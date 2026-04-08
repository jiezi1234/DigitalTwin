"""
教材图片检索技能 (TextbookImageSkill)

供 TutorReActAgent 调用，搜索多模态图片库。
返回图片描述摘要字符串，并将原始图片元数据存入 _last_image_results 供路由层推流。
"""

import logging
from typing import TYPE_CHECKING, List, Dict, Any

from src.skills.base_skill.scripts.base_skill import BaseSkill

if TYPE_CHECKING:
    from src.services.textbook_rag_service import TextbookRAGService

logger = logging.getLogger(__name__)


class TextbookImageSkill(BaseSkill):
    """
    检索教材图片内容（多模态图片向量库）
    """

    name = "search_textbook_images"
    description = (
        "从课本图库中检索与问题相关的图片和图表。"
        "当用户明确要求看图、理解流程图、架构图、E-R 图、SQL 执行过程图等可视化内容时调用。"
        "输入应为简洁的图片主题关键词（例如：'E-R 图'、'关系代数运算示意'、'SQL 执行计划'）。"
        "返回命中图片的摘要说明，同时系统会在最终回答中自动渲染相关图片。"
    )

    def __init__(self, rag_service: "TextbookRAGService", image_k: int = 4):
        self.rag_service = rag_service
        self.image_k = image_k
        # Agent 调用后把序列化图片数据存在这里，供路由层推流
        self._last_image_results: List[Dict[str, Any]] = []

    def run(self, action_input: str) -> str:
        query = action_input.strip()
        if not query:
            return "检索查询不能为空。"

        logger.info(f"[TextbookImageSkill] 图片检索查询: {query}")
        try:
            payload = self.rag_service.retrieve(query, text_k=0, image_k=self.image_k)
            image_results = payload.get("image_results", [])
            self._last_image_results = self.rag_service.serialize_images(image_results)

            if not image_results:
                return "未在教材图库中检索到相关图片。"

            # 返回给 Agent 的文字描述，让 LLM 知道找到了什么图
            context = self.rag_service.format_image_context(image_results, max_items=self.image_k)
            return (
                f"检索到以下相关图片（将在回答中自动渲染，请使用 [图1]、[图2] 等标记引用）：\n{context}"
            )
        except Exception as e:
            logger.error(f"[TextbookImageSkill] 图片检索异常: {e}")
            return f"图片检索时出现异常: {e}"
