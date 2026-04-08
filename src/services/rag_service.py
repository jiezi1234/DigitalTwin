import logging
from typing import List, Dict, Any, Optional, Tuple

from src.infrastructure.llm_client import LLMClient
from src.infrastructure.db_client import DBClient
from src.rag.rag_engine import RAGEngine
from src.rag.query_processor import QueryProcessor
from src.infrastructure.telemetry import get_tracer

from src.skills.chat_history_retrieval.scripts.retrieval_skill import ChatHistoryRetrievalSkill
from src.agent.react_agent import ReActAgent

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

class RAGService:
    """分身 RAG 服务 (集成 Agent 架构)"""

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
        """
        self.llm_client = llm_client
        self.db_client = db_client
        self.collection_name = collection_name

        # 初始化底层核心组件
        self.rag_engine = RAGEngine(db_client=db_client)
        self.query_processor = QueryProcessor(
            llm_client=llm_client,
            enable_coreference_resolution=enable_coreference_resolution,
            enable_query_rewriting=enable_query_rewriting,
        )

        # 封装技能
        self.retrieval_skill = ChatHistoryRetrievalSkill(
            rag_engine=self.rag_engine,
            collection_name=self.collection_name,
            query_processor=self.query_processor,
            k=15,
            lambda_mult=0.6
        )

        # 实例化 Agent
        self.agent = ReActAgent(
            llm_client=self.llm_client,
            tools=[self.retrieval_skill],
            max_iterations=5
        )

    def search(
        self,
        query: str,
        persona: Optional[Dict[str, Any]] = None,
        k: int = 15,
        lambda_mult: float = 0.6,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """搜索相关聊天记录"""
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

    def chat(
        self,
        query: str,
        conversation: List[Dict[str, str]],
        persona: Dict[str, Any],
        max_tokens: int = 500
    ) -> Tuple[str, Dict[str, Any]]:
        """
        基于 Agent 的 RAG 对话逻辑
        """
        with tracer.start_as_current_span("rag_service.chat_agent") as span:
            logger.info(f"[Agent WorkFlow] 新请求汇入: {query}")
            
            # 在启动 agent 时，更新 skill 里的 persona
            self.retrieval_skill.persona = persona

            # 直接交由 ReAct 代理执行
            reply, eval_stats = self.agent.run(
                query=query,
                conversation_history=conversation,
                persona=persona,
                max_tokens=max_tokens
            )
            
            logger.info(f"[Agent Workflow] Session 结束")
            return reply, eval_stats

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.db_client.get_stats(collection_name=self.collection_name)
