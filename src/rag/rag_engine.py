"""
RAG 核心搜索引擎
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from src.infrastructure.db_client import DBClient
from src.rag.bm25_retriever import BM25Retriever
from src.rag.llm_reranker import LLMReranker
from src.rag.query_processor import QueryProcessor
from src.infrastructure.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class RAGEngine:
    """RAG 核心搜索引擎"""

    def __init__(
        self,
        db_client: DBClient,
        lexical_retriever: Optional[BM25Retriever] = None,
        reranker: Optional[LLMReranker] = None,
    ):
        """
        初始化 RAG 引擎

        Args:
            db_client: 数据库客户端
            lexical_retriever: 可选的关键词检索器
            reranker: 可选的候选重排器
        """
        self.db_client = db_client
        self.lexical_retriever = lexical_retriever
        self.reranker = reranker

    def search(
        self,
        query: str,
        collection_name: str,
        query_processor: Optional[QueryProcessor] = None,
        k: int = 15,
        use_mmr: bool = True,
        lambda_mult: float = 0.6,
        hybrid_search: bool = False,
        hybrid_candidates: int = 30,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        rerank: bool = False,
        rerank_candidates: int = 20,
        **kwargs,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        搜索向量数据库

        Args:
            query: 查询文本
            collection_name: 集合名称
            query_processor: 查询处理器（可选）
            k: 返回结果数
            use_mmr: 是否使用 MMR 搜索
            lambda_mult: MMR 多样性权重
            hybrid_search: 是否融合 Dense/MMR 与 BM25 排名
            hybrid_candidates: 每个召回通道进入融合的候选数
            rrf_k: RRF 排名平滑常数
            dense_weight: Dense/MMR 通道权重
            bm25_weight: BM25 通道权重
            rerank: 是否对首轮召回候选执行相关性重排
            rerank_candidates: 送入重排器的候选数
            **kwargs: 其他参数（如 persona 等）

        Returns:
            List of (content, metadata, score)
        """
        with tracer.start_as_current_span("rag.search") as span:
            span.set_attribute("rag.query_original", query[:100])
            span.set_attribute("rag.collection", collection_name)
            span.set_attribute("rag.k", k)
            effective_hybrid_search = (
                hybrid_search and self.lexical_retriever is not None
            )
            effective_reranking = rerank and self.reranker is not None
            span.set_attribute("rag.hybrid_search", effective_hybrid_search)
            span.set_attribute("rag.reranking", effective_reranking)

            try:
                # 处理查询
                processed_query = query
                if query_processor:
                    processed_query = query_processor.process(
                        query,
                        persona=kwargs.get("persona"),
                        conversation=kwargs.get("conversation"),
                    )
                    span.set_attribute("rag.query_processed", processed_query[:100])

                retrieval_limit = (
                    max(k, rerank_candidates) if effective_reranking else k
                )
                candidate_count = max(retrieval_limit, hybrid_candidates)
                dense_error = None
                try:
                    dense_results = self.db_client.search(
                        query=processed_query,
                        collection_name=collection_name,
                        k=(
                            candidate_count
                            if effective_hybrid_search
                            else retrieval_limit
                        ),
                        use_mmr=use_mmr,
                        lambda_mult=lambda_mult,
                    )
                except Exception as exc:
                    if not effective_hybrid_search:
                        raise
                    dense_error = exc
                    dense_results = []
                    logger.warning("Dense/MMR 检索失败，尝试降级到 BM25: %s", exc)

                if effective_hybrid_search:
                    try:
                        bm25_results = self.lexical_retriever.search(
                            query=processed_query,
                            collection_name=collection_name,
                            k=candidate_count,
                        )
                    except Exception as exc:
                        logger.warning("BM25 检索失败，降级到 Dense/MMR: %s", exc)
                        bm25_results = []
                        if dense_error is not None:
                            raise dense_error from exc

                    results = self._weighted_rrf(
                        dense_results=dense_results,
                        bm25_results=bm25_results,
                        limit=retrieval_limit,
                        rrf_k=rrf_k,
                        dense_weight=dense_weight,
                        bm25_weight=bm25_weight,
                    )
                    span.set_attribute("rag.dense_results_count", len(dense_results))
                    span.set_attribute("rag.bm25_results_count", len(bm25_results))
                else:
                    results = dense_results

                if effective_reranking:
                    try:
                        results = self.reranker.rerank(
                            query=processed_query,
                            results=results,
                            top_k=k,
                            candidate_limit=rerank_candidates,
                        )
                    except Exception as exc:
                        logger.warning("候选重排失败，保留首轮召回排名: %s", exc)
                        results = results[:k]

                logger.debug(f"[向量检索] 共 {len(results)} 条结果")
                span.set_attribute("rag.results_count", len(results))

                for i, (content, metadata, score) in enumerate(results, 1):
                    logger.debug(
                        f"[向量检索] #{i} score={score:.4f} | {content.strip()[:150]}"
                    )

                return results

            except Exception as e:
                logger.error(f"RAG 搜索失败: {e}")
                span.record_exception(e)
                raise

    @classmethod
    def _weighted_rrf(
        cls,
        dense_results: List[Tuple[str, Dict[str, Any], float]],
        bm25_results: List[Tuple[str, Dict[str, Any], float]],
        limit: int,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """使用加权 Reciprocal Rank Fusion 合并不同量纲的排名。"""
        rrf_k = max(1, rrf_k)
        dense_weight = max(0.0, dense_weight)
        bm25_weight = max(0.0, bm25_weight)
        if dense_weight == 0.0 and bm25_weight == 0.0:
            dense_weight = 1.0

        fused: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

        def add_channel(
            results: List[Tuple[str, Dict[str, Any], float]],
            channel: str,
            weight: float,
        ) -> None:
            if weight <= 0.0:
                return
            for rank, (content, metadata, raw_score) in enumerate(results, 1):
                key = cls._chat_record_key(content, metadata)
                if key not in fused:
                    fused[key] = {
                        "content": content,
                        "metadata": dict(metadata or {}),
                        "score": 0.0,
                        "channels": [],
                    }
                entry = fused[key]
                entry["score"] += weight / (rrf_k + rank)
                entry["channels"].append(channel)
                entry["metadata"][f"{channel}_score"] = raw_score

        add_channel(dense_results, "dense", dense_weight)
        add_channel(bm25_results, "bm25", bm25_weight)
        ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)

        output = []
        for entry in ranked[: max(0, limit)]:
            metadata = entry["metadata"]
            metadata["retrieval_channels"] = ",".join(entry["channels"])
            output.append((entry["content"], metadata, entry["score"]))
        return output

    def format_context(
        self,
        results: List[Tuple[str, Dict[str, Any], float]],
        max_context_length: int = 2000,
        include_metadata: bool = True,
        format_type: str = "chat",  # "chat" 或 "textbook"
    ) -> str:
        """
        格式化搜索结果为上下文字符串

        Args:
            results: 搜索结果列表
            max_context_length: 最大上下文长度
            include_metadata: 是否包含元数据
            format_type: 格式化类型（chat 或 textbook）

        Returns:
            格式化的上下文字符串
        """
        with tracer.start_as_current_span("format.context") as span:
            span.set_attribute("format.type", format_type)
            span.set_attribute("format.num_results", len(results))

            if not results:
                return ""

            lines = []
            total_length = 0

            for content, metadata, score in results:
                if format_type == "chat":
                    # 聊天记录格式
                    if include_metadata:
                        talker = metadata.get("talker", "未知")
                        chat_time = metadata.get("chat_time_str") or metadata.get(
                            "chat_time", ""
                        )
                        time_prefix = f"[{chat_time}] " if chat_time else ""
                        record = f"{time_prefix}{talker}: {content.strip()}"
                    else:
                        record = content.strip()

                elif format_type == "textbook":
                    # 教材格式（带编号，供 LLM 引用）
                    idx = len(lines) + 1
                    if include_metadata:
                        source_file = metadata.get("source_file", "")
                        chapter = metadata.get("chapter", "")
                        section = metadata.get("section", "")
                        page = metadata.get("page", "")

                        location_parts = []
                        if source_file:
                            location_parts.append(source_file)
                        if chapter:
                            location_parts.append(chapter)
                        if section:
                            location_parts.append(section)
                        if page:
                            location_parts.append(f"第{page}页")

                        location = " > ".join(location_parts) if location_parts else ""
                        record = f"[{idx}]【{location}】\n{content.strip()}\n"
                    else:
                        record = f"[{idx}] {content.strip()}"

                else:
                    # 默认格式
                    record = content.strip()

                if total_length + len(record) > max_context_length:
                    break

                lines.append(record)
                total_length += len(record)

            return "\n".join(lines)

    @staticmethod
    def _chat_record_key(content: str, metadata: Dict[str, Any]) -> Tuple[Any, ...]:
        conversation_id = metadata.get("conversation_id")
        message_index = metadata.get("message_index")
        if conversation_id and message_index is not None:
            return ("conversation", conversation_id, str(message_index))
        return (
            "legacy",
            metadata.get("source_file", ""),
            metadata.get("chat_time", ""),
            content.strip(),
        )

    def expand_chat_neighbors(
        self,
        results: List[Tuple[str, Dict[str, Any], float]],
        collection_name: str,
        window_size: int = 2,
        anchor_limit: int = 5,
        max_results: int = 30,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """围绕语义命中消息补齐同一会话的前后消息。"""
        max_results = max(1, max_results)
        if not results or window_size <= 0 or anchor_limit <= 0:
            return results[:max_results]

        with tracer.start_as_current_span("rag.expand_chat_neighbors") as span:
            expanded: List[Tuple[str, Dict[str, Any], float]] = []
            seen = set()

            def append_record(
                content: str,
                metadata: Dict[str, Any],
                score: float,
                origin: str,
                anchor_rank: int,
                distance: int = 0,
            ) -> None:
                key = self._chat_record_key(content, metadata)
                if key in seen or len(expanded) >= max_results:
                    return
                enriched_metadata = dict(metadata or {})
                enriched_metadata["retrieval_origin"] = origin
                enriched_metadata["anchor_rank"] = anchor_rank
                enriched_metadata["neighbor_distance"] = distance
                expanded.append((content, enriched_metadata, score))
                seen.add(key)

            for anchor_rank, (content, metadata, score) in enumerate(
                results[:anchor_limit], 1
            ):
                conversation_id = metadata.get("conversation_id")
                try:
                    message_index = int(metadata.get("message_index"))
                except (TypeError, ValueError):
                    append_record(content, metadata, score, "semantic", anchor_rank)
                    continue

                if not conversation_id:
                    append_record(content, metadata, score, "semantic", anchor_rank)
                    continue

                where = {
                    "$and": [
                        {"conversation_id": {"$eq": str(conversation_id)}},
                        {
                            "message_index": {
                                "$gte": max(0, message_index - window_size)
                            }
                        },
                        {"message_index": {"$lte": message_index + window_size}},
                    ]
                }
                try:
                    neighbors = self.db_client.get_records(
                        collection_name=collection_name,
                        where=where,
                        limit=window_size * 2 + 1,
                    )
                except Exception as exc:
                    logger.warning("聊天邻域读取失败，保留语义命中结果: %s", exc)
                    neighbors = []

                if not neighbors:
                    append_record(content, metadata, score, "semantic", anchor_rank)
                    continue

                def message_order(item: Tuple[str, Dict[str, Any], float]) -> int:
                    try:
                        return int(item[1].get("message_index", message_index))
                    except (TypeError, ValueError):
                        return message_index

                neighbors.sort(key=message_order)
                for neighbor_content, neighbor_metadata, _ in neighbors:
                    try:
                        neighbor_index = int(neighbor_metadata.get("message_index"))
                    except (TypeError, ValueError):
                        neighbor_index = message_index
                    distance = abs(neighbor_index - message_index)
                    origin = "semantic" if distance == 0 else "neighbor"
                    derived_score = max(0.0, score * (1.0 - 0.05 * distance))
                    append_record(
                        neighbor_content,
                        neighbor_metadata,
                        derived_score,
                        origin,
                        anchor_rank,
                        distance,
                    )

            for semantic_rank, (content, metadata, score) in enumerate(results, 1):
                append_record(content, metadata, score, "semantic", semantic_rank)

            neighbor_count = sum(
                1
                for _, metadata, _ in expanded
                if metadata.get("retrieval_origin") == "neighbor"
            )
            span.set_attribute("rag.semantic_results", len(results))
            span.set_attribute("rag.expanded_results", len(expanded))
            span.set_attribute("rag.neighbor_results", neighbor_count)
            return expanded
