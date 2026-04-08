"""
教材文本检索技能 (TextbookTextSkill)

供 TutorReActAgent 调用，同时搜索多模态文本库和 OCR 文本库，
返回格式化后的文本段落字符串（含来源标注）。
"""

import logging
from typing import TYPE_CHECKING

from src.skills.base_skill.scripts.base_skill import BaseSkill

if TYPE_CHECKING:
    from src.services.textbook_rag_service import TextbookRAGService

logger = logging.getLogger(__name__)


class TextbookTextSkill(BaseSkill):
    """
    检索教材文本内容（多模态文本块 + OCR 扫描文字双路召回）
    """

    name = "search_textbook_text"
    description = (
        "从课本资料（课件 PPT 和扫描版教材）中检索相关文字段落。"
        "当用户的问题涉及概念解释、定义、知识点、SQL 语句、理论推导等文字内容时调用。"
        "输入应为简洁的查询关键词或问题（例如：'SQL 定义'、'第三范式的含义'）。"
        "返回匹配的文字段落列表，带有来源和页码标注。"
    )

    def __init__(self, rag_service: "TextbookRAGService", text_k: int = 8):
        self.rag_service = rag_service
        self.text_k = text_k
        # Agent 调用后把原始结果存在这里，供路由层提取 sources
        self._last_results = []

    def run(self, action_input: str) -> str:
        query = action_input.strip()
        if not query:
            return "检索查询不能为空。"

        logger.info(f"[TextbookTextSkill] 检索查询: {query}")
        try:
            payload = self.rag_service.retrieve(query, text_k=self.text_k, image_k=0)
            results = payload.get("text_results", [])
            self._last_results = results

            if not results:
                return "未在教材中检索到相关文字内容。"

            context = self.rag_service.format_context(
                results,
                max_context_length=3000,
                include_metadata=True,
            )
            return f"检索到以下教材段落：\n{context}"
        except Exception as e:
            logger.error(f"[TextbookTextSkill] 检索异常: {e}")
            return f"文本检索时出现异常: {e}"
