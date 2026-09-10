"""
查询处理器
支持指代消解、Query Rewriting 等处理策略
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Literal
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)
_JSON_OBJECT = re.compile(r"\{[\s\S]*\}")


@dataclass(frozen=True)
class QueryUnderstanding:
    """结构化查询理解结果，供检索与后续元数据过滤复用。"""

    original_query: str
    standalone_query: str
    entities: List[str] = field(default_factory=list)
    time_range: Optional[Dict[str, str]] = None


class QueryProcessor:
    """查询处理器，支持多种处理策略"""

    def __init__(
        self,
        llm_client: LLMClient,
        enable_coreference_resolution: bool = True,
        enable_query_rewriting: bool = True,
        domain: Literal["persona", "textbook"] = "persona",
        history_messages: int = 6,
    ):
        """
        初始化查询处理器

        Args:
            llm_client: LLM 客户端
            enable_coreference_resolution: 是否启用指代消解
            enable_query_rewriting: 是否启用查询改写
            domain: 查询所属领域，人物对话或教材知识库
            history_messages: 查询理解读取的最近会话消息数
        """
        self.llm_client = llm_client
        self.enable_coreference_resolution = enable_coreference_resolution
        self.enable_query_rewriting = enable_query_rewriting
        self.domain = domain
        self.history_messages = max(0, history_messages)

    def _format_history(
        self,
        conversation: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, str]]:
        """截取并清洗最近会话，避免把无限历史注入查询理解提示词。"""
        if not conversation or self.history_messages == 0:
            return []

        history: List[Dict[str, str]] = []
        for message in conversation[-self.history_messages :]:
            role = str(message.get("role", "unknown"))
            content = str(message.get("content", "")).strip()
            if content:
                history.append({"role": role, "content": content[:500]})
        return history

    def _build_understanding_prompt(
        self,
        query: str,
        persona: Optional[Dict[str, Any]],
        conversation: Optional[List[Dict[str, Any]]],
    ) -> str:
        history = self._format_history(conversation)
        task_parts = []
        if self.enable_coreference_resolution:
            task_parts.append("结合最近会话消解代词、省略和上下文指代")
        if self.enable_query_rewriting:
            task_parts.append("将问题改写为适合语义检索的独立查询")

        if self.domain == "textbook":
            domain_instruction = (
                "场景是课程教材问答。保留专业实体、约束条件、页码以及中英文术语，"
                "可以补充必要的同义词或上位概念，但不要回答问题。"
            )
            persona_context: Dict[str, Any] = {}
        else:
            domain_instruction = (
                "场景是人物聊天记忆检索。保留人物、事件、地点、时间和口语表达，"
                "不要把未知事实补写进查询。"
            )
            active_persona = persona or {}
            persona_context = {
                "name": active_persona.get("name", ""),
                "system_prompt": str(active_persona.get("system_prompt", ""))[:200],
            }

        payload = {
            "domain": self.domain,
            "persona": persona_context,
            "conversation": history,
            "query": query,
        }
        return (
            "你是RAG系统的查询理解模块。"
            + "；".join(task_parts)
            + "。\n"
            + domain_instruction
            + "\n只输出合法JSON，不要解释，不要输出Markdown代码块。格式为：\n"
            + '{"standalone_query":"...","entities":["..."],'
            + '"time_range":{"start":"...","end":"..."}}\n'
            + "无法确定时间范围时将time_range设为null；没有实体时entities为空数组。\n"
            + f"输入：{json.dumps(payload, ensure_ascii=False)}"
        )

    @staticmethod
    def _parse_understanding(query: str, raw: Optional[str]) -> QueryUnderstanding:
        """解析结构化结果；兼容旧的纯文本改写响应。"""
        text = (raw or "").strip()
        if not text:
            return QueryUnderstanding(original_query=query, standalone_query=query)

        match = _JSON_OBJECT.search(text)
        if match:
            try:
                payload = json.loads(match.group(0))
                standalone_query = (
                    str(payload.get("standalone_query", "")).strip() or query
                )
                raw_entities = payload.get("entities", [])
                entities = (
                    [str(item).strip() for item in raw_entities if str(item).strip()]
                    if isinstance(raw_entities, list)
                    else []
                )
                raw_time_range = payload.get("time_range")
                time_range = None
                if isinstance(raw_time_range, dict):
                    cleaned = {
                        str(key): str(value).strip()
                        for key, value in raw_time_range.items()
                        if value is not None and str(value).strip()
                    }
                    time_range = cleaned or None
                return QueryUnderstanding(
                    original_query=query,
                    standalone_query=standalone_query,
                    entities=entities,
                    time_range=time_range,
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                logger.warning("查询理解响应不是合法JSON，回退到纯文本结果")

        return QueryUnderstanding(original_query=query, standalone_query=text)

    def understand(
        self,
        query: str,
        persona: Optional[Dict[str, Any]] = None,
        conversation: Optional[List[Dict[str, Any]]] = None,
    ) -> QueryUnderstanding:
        """用一次模型调用完成历史感知的指代消解与查询改写。"""
        if not self.enable_coreference_resolution and not self.enable_query_rewriting:
            return QueryUnderstanding(original_query=query, standalone_query=query)

        with tracer.start_as_current_span("query.understand") as span:
            span.set_attribute("query.domain", self.domain)
            span.set_attribute(
                "query.history_messages", len(self._format_history(conversation))
            )
            try:
                raw = self.llm_client.call(
                    messages=[
                        {
                            "role": "user",
                            "content": self._build_understanding_prompt(
                                query, persona, conversation
                            ),
                        }
                    ],
                    temperature=0.0,
                    max_tokens=300,
                )
                result = self._parse_understanding(query, raw)
                span.set_attribute("query.changed", result.standalone_query != query)
                span.set_attribute("query.entities_count", len(result.entities))
                return result
            except Exception as exc:
                logger.warning("历史感知查询理解失败，回退到原查询: %s", exc)
                span.record_exception(exc)
                return QueryUnderstanding(original_query=query, standalone_query=query)

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
                    logger.debug(f"[指代消解] {query} → {result}")
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
            if self.domain == "textbook":
                prompt = f"""你的任务是改写教材问答查询，使其更容易从课程教材中检索相关内容。

保留原问题的专业实体、约束条件和术语，并补充必要的同义词或上位概念。
例如：
- "ACID是什么？" 可改写为：数据库事务 ACID 原子性 一致性 隔离性 持久性
- "怎么建索引？" 可改写为：数据库索引 创建索引 CREATE INDEX 使用方法

原问题：{query}

请直接输出适合语义检索的查询，不要回答问题，不要添加说明。"""
            else:
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
                    logger.debug(f"[查询重写] {query} → {result}")
                    span.set_attribute("query.rewritten", True)
                    return result

            except Exception as e:
                logger.warning(f"Query改写失败: {e}")
                span.record_exception(e)

            return query

    def process(
        self,
        query: str,
        persona: Optional[Dict[str, Any]] = None,
        conversation: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        处理查询（完整流程）

        Args:
            query: 原始查询
            persona: 分身信息
            conversation: 最近会话历史

        Returns:
            处理后的查询
        """
        with tracer.start_as_current_span("query.process") as span:
            span.set_attribute("query.original", query[:100])

            result = self.understand(
                query=query,
                persona=persona,
                conversation=conversation,
            )
            span.set_attribute("query.processed", result.standalone_query[:100])
            return result.standalone_query
