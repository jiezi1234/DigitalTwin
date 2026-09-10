import logging
from typing import List, Dict, Any, Optional, Tuple
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.db_client import DBClient
from src.rag.rag_engine import RAGEngine
from src.rag.bm25_retriever import BM25Retriever
from src.rag.llm_reranker import LLMReranker
from src.rag.metadata_filter import MetadataFilterBuilder
from src.rag.query_processor import QueryProcessor
from src.rag.react_router import ReActRetrievalRouter
from src.infrastructure.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class RAGService:
    """分身 RAG 服务 (集成架构)"""

    def __init__(
        self,
        llm_client: LLMClient,
        db_client: DBClient,
        collection_name: str = "wechat_embeddings",
        enable_coreference_resolution: bool = True,
        enable_query_rewriting: bool = True,
        query_history_messages: int = 6,
        enable_neighbor_expansion: bool = True,
        neighbor_window: int = 2,
        neighbor_anchors: int = 5,
        neighbor_max_results: int = 30,
        enable_hybrid_search: bool = False,
        hybrid_candidates: int = 30,
        hybrid_rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        bm25_retriever: Optional[BM25Retriever] = None,
        enable_reranking: bool = False,
        rerank_model: str = "qwen-turbo",
        rerank_candidates: int = 20,
        reranker: Optional[LLMReranker] = None,
        enable_metadata_filtering: bool = False,
        timezone_offset: str = "+08:00",
        react_router: Optional[ReActRetrievalRouter] = None,
        retrieval_enabled: bool = True,
        max_results: int = 15,
        max_context_length: int = 2000,
        include_metadata: bool = True,
    ):
        """
        初始化 RAG 服务

        Args:
            llm_client: LLM 客户端
            db_client: 数据库客户端
            collection_name: 向量集合名称
            enable_coreference_resolution: 启用指代消解
            enable_query_rewriting: 启用 Query Rewriting
            query_history_messages: 查询理解读取的最近会话消息数
            enable_neighbor_expansion: 是否补齐语义命中消息的时间邻域
            neighbor_window: 每个语义命中点前后补充的消息数
            neighbor_anchors: 执行邻域扩展的语义命中点数量
            neighbor_max_results: 邻域扩展后的最大记录数
            enable_hybrid_search: 是否启用 Dense/MMR + BM25 混合召回
            hybrid_candidates: 每个召回通道参与 RRF 融合的候选数
            hybrid_rrf_k: RRF 排名平滑常数
            dense_weight: Dense/MMR 通道融合权重
            bm25_weight: BM25 通道融合权重
            bm25_retriever: 可注入的 BM25 检索器，主要用于测试或复用索引
            enable_reranking: 是否对召回候选执行 LLM 相关性重排
            rerank_model: 重排使用的模型
            rerank_candidates: 送入重排器的候选数
            reranker: 可注入的候选重排器
            enable_metadata_filtering: 是否应用查询理解产生的时间约束
            timezone_offset: 无时区日期采用的 UTC 偏移
            react_router: 可选的 ReAct 检索工具路由器
            retrieval_enabled: 是否允许调用检索工具
            max_results: 单次检索的最大结果数
            max_context_length: 注入提示词的最大上下文字符数
            include_metadata: 检索上下文是否包含来源元数据
        """
        self.llm_client = llm_client
        self.db_client = db_client
        self.collection_name = collection_name
        self.react_router = react_router
        self.retrieval_enabled = retrieval_enabled
        self.max_results = max(1, max_results)
        self.max_context_length = max(1, max_context_length)
        self.include_metadata = include_metadata
        self.enable_neighbor_expansion = enable_neighbor_expansion
        self.neighbor_window = max(0, neighbor_window)
        self.neighbor_anchors = max(0, neighbor_anchors)
        self.neighbor_max_results = max(1, neighbor_max_results)
        self.enable_hybrid_search = enable_hybrid_search
        self.hybrid_candidates = max(1, hybrid_candidates)
        self.hybrid_rrf_k = max(1, hybrid_rrf_k)
        self.dense_weight = max(0.0, dense_weight)
        self.bm25_weight = max(0.0, bm25_weight)
        if self.dense_weight == 0.0 and self.bm25_weight == 0.0:
            self.dense_weight = 1.0
        self.enable_reranking = enable_reranking
        self.rerank_candidates = max(1, rerank_candidates)
        self.enable_metadata_filtering = enable_metadata_filtering

        # 初始化核心组件
        lexical_retriever = bm25_retriever
        if self.enable_hybrid_search and lexical_retriever is None:
            lexical_retriever = BM25Retriever(db_client=db_client)
        active_reranker = reranker
        if self.enable_reranking and active_reranker is None:
            active_reranker = LLMReranker(
                llm_client=llm_client,
                model=rerank_model,
            )
        metadata_filter_builder = (
            MetadataFilterBuilder(timezone_offset=timezone_offset)
            if self.enable_metadata_filtering
            else None
        )
        self.rag_engine = RAGEngine(
            db_client=db_client,
            lexical_retriever=lexical_retriever,
            reranker=active_reranker,
            metadata_filter_builder=metadata_filter_builder,
        )
        self.query_processor = QueryProcessor(
            llm_client=llm_client,
            enable_coreference_resolution=enable_coreference_resolution,
            enable_query_rewriting=enable_query_rewriting,
            history_messages=query_history_messages,
        )

    def search(
        self,
        query: str,
        persona: Optional[Dict[str, Any]] = None,
        conversation: Optional[List[Dict[str, Any]]] = None,
        k: int = 15,
        lambda_mult: float = 0.6,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """搜索相关聊天记录"""
        results, _ = self._search_with_neighbors(
            query=query,
            persona=persona,
            conversation=conversation,
            k=k,
            lambda_mult=lambda_mult,
        )
        return results

    def _search_with_neighbors(
        self,
        query: str,
        persona: Optional[Dict[str, Any]],
        conversation: Optional[List[Dict[str, Any]]],
        k: int,
        lambda_mult: float = 0.6,
    ) -> Tuple[List[Tuple[str, Dict[str, Any], float]], int]:
        semantic_results = self.rag_engine.search(
            query=query,
            collection_name=self.collection_name,
            query_processor=self.query_processor,
            k=k,
            lambda_mult=lambda_mult,
            hybrid_search=self.enable_hybrid_search,
            hybrid_candidates=self.hybrid_candidates,
            rrf_k=self.hybrid_rrf_k,
            dense_weight=self.dense_weight,
            bm25_weight=self.bm25_weight,
            rerank=self.enable_reranking,
            rerank_candidates=self.rerank_candidates,
            metadata_filtering=self.enable_metadata_filtering,
            persona=persona,
            conversation=conversation,
        )
        if not self.enable_neighbor_expansion:
            return semantic_results, len(semantic_results)

        expanded_results = self.rag_engine.expand_chat_neighbors(
            semantic_results,
            collection_name=self.collection_name,
            window_size=self.neighbor_window,
            anchor_limit=self.neighbor_anchors,
            max_results=self.neighbor_max_results,
        )
        return expanded_results, len(semantic_results)

    def format_context(
        self,
        results: List[Tuple[str, Dict[str, Any], float]],
        max_context_length: int = 2000,
        include_metadata: bool = True,
    ) -> str:
        """格式化搜索结果"""
        return self.rag_engine.format_context(
            results,
            max_context_length=max_context_length,
            include_metadata=include_metadata,
            format_type="chat",
        )

    def chat(
        self,
        query: str,
        conversation: List[Dict[str, str]],
        persona: Dict[str, Any],
        system_prefix: str = "相关记录：\n",
        role_instruction: str = "",
        max_tokens: int = 500,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        标准 RAG 对话逻辑：查询处理、检索、上下文构建与回答生成。

        Returns:
            (回复文本, 基础检索元数据)
        """
        with tracer.start_as_current_span("rag_service.chat") as span:
            logger.debug(f"[用户输入] {query}")
            action = "retrieve"
            route_mode = "standard"
            if not self.retrieval_enabled:
                action = "respond"
                route_mode = "disabled"
            elif self.react_router:
                decision = self.react_router.decide(
                    query=query,
                    conversation=conversation,
                    persona=persona,
                )
                action = decision.action
                route_mode = "react"

            # retrieval_search 是当前 ReAct 代理唯一的外部工具。
            semantic_result_count = 0
            if action == "retrieve":
                results, semantic_result_count = self._search_with_neighbors(
                    query,
                    persona=persona,
                    conversation=conversation,
                    k=self.max_results,
                )
            else:
                results = []
            context_text = (
                self.format_context(
                    results,
                    max_context_length=self.max_context_length,
                    include_metadata=self.include_metadata,
                )
                if results
                else ""
            )
            retrieval_stats = {
                "route_mode": route_mode,
                "action": action,
                "retrieved": action == "retrieve",
                "retrieval_mode": ("hybrid" if self.enable_hybrid_search else "dense"),
                "reranking_enabled": self.enable_reranking,
                "reranked": any(
                    "rerank_score" in metadata for _, metadata, _ in results
                ),
                "metadata_filtering_enabled": self.enable_metadata_filtering,
                "metadata_filtered": any(
                    metadata.get("metadata_filter_applied", False)
                    for _, metadata, _ in results
                ),
                "result_count": len(results),
                "semantic_result_count": semantic_result_count,
                "neighbor_count": sum(
                    1
                    for _, metadata, _ in results
                    if metadata.get("retrieval_origin") == "neighbor"
                ),
            }
            span.set_attribute("rag.route_mode", route_mode)
            span.set_attribute("rag.action", action)

            reply = self._generate(
                query,
                conversation,
                persona,
                context_text,
                system_prefix,
                role_instruction,
                max_tokens,
            )

            if not reply:
                return "抱歉，我目前无法回答这个问题。", retrieval_stats

            logger.debug(f"[最终输出] {reply}")
            return reply, retrieval_stats

    def _generate(
        self,
        query: str,
        conversation: List[Dict[str, str]],
        persona: Dict[str, Any],
        context_text: str,
        system_prefix: str,
        role_instruction: str,
        max_tokens: int,
    ) -> Optional[str]:
        """
        构建 prompt 并调用 LLM 生成回复。

        Returns:
            回复文本；模型调用失败时返回 None。
        """
        full_system = f"{persona['system_prompt']}\n\n{role_instruction}"
        if context_text:
            full_system = f"{system_prefix}{context_text}\n\n{full_system}"

        messages = [{"role": "system", "content": full_system}]
        messages.extend(conversation)
        messages.append({"role": "user", "content": query})

        return self.llm_client.call(messages, max_tokens=max_tokens)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.db_client.get_stats(collection_name=self.collection_name)
