"""
简化后的 RAG 服务（分身专用）
调用 RAGEngine 和 QueryProcessor
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.db_client import DBClient
from src.rag.rag_engine import RAGEngine
from src.rag.query_processor import QueryProcessor

logger = logging.getLogger(__name__)


class RAGService:
    """分身 RAG 服务（简化版本）"""

    def __init__(
        self,
        llm_client: LLMClient,
        db_client: DBClient,
        collection_name: str = "wechat_embeddings",
        enable_coreference_resolution: bool = True,
        enable_query_rewriting: bool = True,
    ):
        """
        初始化 RAG 服务

        Args:
            llm_client: LLM 客户端
            db_client: 数据库客户端
            collection_name: 向量集合名称
            enable_coreference_resolution: 启用指代消解
            enable_query_rewriting: 启用 Query Rewriting
        """
        self.llm_client = llm_client
        self.db_client = db_client
        self.collection_name = collection_name

        # 初始化核心组件
        self.rag_engine = RAGEngine(db_client=db_client)
        self.query_processor = QueryProcessor(
            llm_client=llm_client,
            enable_coreference_resolution=enable_coreference_resolution,
            enable_query_rewriting=enable_query_rewriting,
        )

    def search(
        self,
        query: str,
        persona: Optional[Dict[str, Any]] = None,
        k: int = 15,
        lambda_mult: float = 0.6,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        搜索相关聊天记录

        Args:
            query: 查询文本
            persona: 分身信息
            k: 返回结果数
            lambda_mult: MMR 多样性权重

        Returns:
            List of (content, metadata, score)
        """
        return self.rag_engine.search(
            query=query,
            collection_name=self.collection_name,
            query_processor=self.query_processor,
            k=k,
            lambda_mult=lambda_mult,
            persona=persona,
        )

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

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.db_client.get_stats(collection_name=self.collection_name)
