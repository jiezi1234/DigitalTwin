"""
查询处理器
支持指代消解、Query Rewriting 等处理策略
"""

import logging
from typing import Optional, Dict, Any, List
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class QueryProcessor:
    """查询处理器，支持多种处理策略"""

    def __init__(
        self,
        llm_client: LLMClient,
        enable_coreference_resolution: bool = True,
        enable_query_rewriting: bool = True,
    ):
        """
        初始化查询处理器

        Args:
            llm_client: LLM 客户端
            enable_coreference_resolution: 是否启用指代消解
            enable_query_rewriting: 是否启用查询改写
        """
        self.llm_client = llm_client
        self.enable_coreference_resolution = enable_coreference_resolution
        self.enable_query_rewriting = enable_query_rewriting

    def resolve_coreference(
        self, query: str, persona: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        指代消解：将代词替换为具体的人名或概念

        Args:
            query: 原始查询
            persona: 分身信息（包含名字等上下文）

        Returns:
            消解后的查询
        """
        if not self.enable_coreference_resolution:
            return query

        with tracer.start_as_current_span("query.coreference_resolution") as span:
            # 检查是否包含常见代词
            pronouns = ["他", "她", "它", "他们", "她们", "它们", "那个", "这个"]
            if not any(p in query for p in pronouns):
                logger.debug("查询中无代词，跳过消解")
                return query

            persona_name = (persona or {}).get("name", "")
            persona_info = f"分身名字：{persona_name}\n" if persona_name else ""

            prompt = f"""{persona_info}你的任务是进行指代消解（Coreference Resolution）。

将下面问题中的代词替换为具体的人名或概念，使问题更清楚。
代词包括：他、她、它、他们、她们、它们、那个、这个等。

如果代词指代不明确或根本不需要替换，保持原样。

原问题：{query}

请直接输出消解后的问题，不要解释。"""

            try:
                result = self.llm_client.call(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=200,
                )

                if result and result != query:
                    logger.info(f"指代消解: '{query}' → '{result}'")
                    span.set_attribute("query.coreference_changed", True)
                    return result

            except Exception as e:
                logger.warning(f"指代消解失败: {e}")
                span.record_exception(e)

            return query

    def rewrite_query(
        self, query: str, persona: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Query Rewriting：根据分身特点改写查询以提高检索质量

        Args:
            query: 原始查询（可能已消解代词）
            persona: 分身信息（包含名字、特点等）

        Returns:
            改写后的查询
        """
        if not self.enable_query_rewriting:
            return query

        with tracer.start_as_current_span("query.rewriting") as span:
            persona_name = (persona or {}).get("name", "")
            system_prompt = (persona or {}).get("system_prompt", "")
            doc_count = (persona or {}).get("doc_count", 0)

            persona_context = f"""分身信息：
- 名字：{persona_name}
- 已导入聊天记录数：{doc_count}条
- 角色设定：{system_prompt[:200] if system_prompt else "未设定"}"""

            prompt = f"""{persona_context}

你的任务是改写用户的问题，使其更容易从分身的聊天历史中检索相关内容。

原问题可能很短或表述模糊，你需要基于分身的特点和背景，将其扩展和转化为更有语义的形式。

例如：
- "你怎么样？" 对于林黛玉可能改写为：身体状况、健康、精神状态、情绪、病症
- "最近在做什么？" 可能改写为：近期活动、日常事务、工作、业余爱好

原问题：{query}

请输出改写后的问题或关键词组合（用中文逗号分隔），使其更适合向量检索。
不要添加额外说明，直接输出改写结果。"""

            try:
                result = self.llm_client.call(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=300,
                )

                if result and result != query:
                    logger.info(f"Query改写: '{query}' → '{result}'")
                    span.set_attribute("query.rewritten", True)
                    return result

            except Exception as e:
                logger.warning(f"Query改写失败: {e}")
                span.record_exception(e)

            return query

    def process(
        self, query: str, persona: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        处理查询（完整流程）

        Args:
            query: 原始查询
            persona: 分身信息

        Returns:
            处理后的查询
        """
        with tracer.start_as_current_span("query.process") as span:
            span.set_attribute("query.original", query[:100])

            # 步骤 1：指代消解
            if self.enable_coreference_resolution:
                query = self.resolve_coreference(query, persona)
                logger.debug(f"指代消解后: {query[:100]}")

            # 步骤 2：Query Rewriting
            if self.enable_query_rewriting:
                query = self.rewrite_query(query, persona)
                logger.debug(f"改写后: {query[:100]}")

            span.set_attribute("query.processed", query[:100])
            return query
