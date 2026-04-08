import logging
from typing import Any, Dict, Optional

from src.skills.base_skill.scripts.base_skill import BaseSkill
from src.rag.rag_engine import RAGEngine
from src.rag.query_processor import QueryProcessor

logger = logging.getLogger(__name__)

class ChatHistoryRetrievalSkill(BaseSkill):
    """
    RAG 搜索技能
    """
    name = "retrieve_chat_history"
    description = (
        "当需要了解用户的历史聊天记录、某些人的背景、过去发生的事情或讨论过的话题时，使用此工具。"
        "输入应当是一个简洁的查询语句或关键词（例如：'运动'，'昨天聊了什么'）。"
        "返回结果为匹配的聊天记录列表。"
    )

    def __init__(
        self, 
        rag_engine: RAGEngine, 
        collection_name: str,
        query_processor: Optional[QueryProcessor] = None,
        persona: Optional[Dict[str, Any]] = None,
        k: int = 15,
        lambda_mult: float = 0.6
    ):
        self.rag_engine = rag_engine
        self.collection_name = collection_name
        self.query_processor = query_processor
        self.persona = persona
        self.k = k
        self.lambda_mult = lambda_mult

    def run(self, action_input: str) -> str:
        query = action_input.strip()
        if not query:
            return "检索查询不能为空。"
            
        logger.info(f"[RetrievalSkill] 检索查询: {query}")
        try:
            results = self.rag_engine.search(
                query=query,
                collection_name=self.collection_name,
                query_processor=self.query_processor,
                k=self.k,
                lambda_mult=self.lambda_mult,
                persona=self.persona
            )
            if not results:
                return "未检索到相关的历史记录。"

            # 使用 format_context 转化为文本
            context_string = self.rag_engine.format_context(
                results, 
                max_context_length=3000, 
                include_metadata=True, 
                format_type="chat"
            )
            return f"检索到的记录如下：\n{context_string}"
        except Exception as e:
            logger.error(f"[RetrievalSkill] 检索出现异常: {e}")
            return f"检索时出现异常: {e}"
