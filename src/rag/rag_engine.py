"""
RAG 核心搜索引擎
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from src.infrastructure.db_client import DBClient
from src.rag.query_processor import QueryProcessor
from src.infrastructure.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class RAGEngine:
    """RAG 核心搜索引擎"""

    def __init__(self, db_client: DBClient):
        """
        初始化 RAG 引擎

        Args:
            db_client: 数据库客户端
        """
        self.db_client = db_client

    def search(
        self,
        query: str,
        collection_name: str,
        query_processor: Optional[QueryProcessor] = None,
        k: int = 15,
        use_mmr: bool = True,
        lambda_mult: float = 0.6,
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
            **kwargs: 其他参数（如 persona 等）

        Returns:
            List of (content, metadata, score)
        """
        with tracer.start_as_current_span("rag.search") as span:
            span.set_attribute("rag.query_original", query[:100])
            span.set_attribute("rag.collection", collection_name)
            span.set_attribute("rag.k", k)

            try:
                # 处理查询
                processed_query = query
                if query_processor:
                    processed_query = query_processor.process(query, persona=kwargs.get("persona"))
                    span.set_attribute("rag.query_processed", processed_query[:100])

                # 向量搜索
                results = self.db_client.search(
                    query=processed_query,
                    collection_name=collection_name,
                    k=k,
                    use_mmr=use_mmr,
                    lambda_mult=lambda_mult,
                )

                logger.debug(f"[向量检索] 共 {len(results)} 条结果")
                span.set_attribute("rag.results_count", len(results))

                for i, (content, metadata, score) in enumerate(results, 1):
                    logger.debug(f"[向量检索] #{i} score={score:.4f} | {content.strip()[:150]}")

                return results

            except Exception as e:
                logger.error(f"RAG 搜索失败: {e}")
                span.record_exception(e)
                raise

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
                        chat_time = metadata.get("chat_time_str") or metadata.get("chat_time", "")
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

    def get_nearby_records(
        self,
        collection_name: str,
        timestamp: int,
        time_window_minutes: int = 30,
        max_nearby: int = 15,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        获取时间戳相近的记录（用于聊天记录）

        Args:
            collection_name: 集合名称
            timestamp: 目标 Unix 时间戳（整数秒）
            time_window_minutes: 时间窗口(分钟)
            max_nearby: 最多返回的相近记录数

        Returns:
            List of (content, metadata, score)
        """
        # 这个方法用于从聊天记录中检索时间相近的记录
        # 目前由于 ChromaDB 的限制，实现较复杂，暂不在引擎层实现
        # 而是在具体的 RAGService 中实现
        logger.debug("获取相近记录功能由具体的 Service 实现")
        return []
