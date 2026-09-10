"""面向检索工具的轻量 ReAct 路由器。"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from src.infrastructure.llm_client import LLMClient
from src.infrastructure.telemetry import get_meter, get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)
meter = get_meter(__name__)

react_routes_total = meter.create_counter(
    "react_routes_total",
    description="Number of ReAct retrieval routing decisions",
    unit="1",
)

_JSON_OBJECT = re.compile(r"\{[\s\S]*?\}")


@dataclass(frozen=True)
class ReActDecision:
    """路由器输出，只暴露动作，不保留模型思维过程。"""

    action: Literal["retrieve", "respond"]


class ReActRetrievalRouter:
    """让模型在调用检索工具和直接回答之间做一次动作选择。"""

    def __init__(
        self,
        llm_client: LLMClient,
        model: str = "qwen-turbo",
        history_messages: int = 6,
        fallback_action: Literal["retrieve", "respond"] = "retrieve",
    ):
        self.llm_client = llm_client
        self.model = model
        self.history_messages = max(0, history_messages)
        self.fallback_action = fallback_action

    def decide(
        self,
        query: str,
        conversation: Optional[List[Dict[str, Any]]] = None,
        persona: Optional[Dict[str, Any]] = None,
    ) -> ReActDecision:
        """选择下一步动作；解析失败时默认检索以降低无依据回答风险。"""
        with tracer.start_as_current_span("react.route") as span:
            prompt = self._build_prompt(query, conversation or [], persona or {})
            raw = self.llm_client.call(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=60,
                model=self.model,
            )
            decision = self._parse_decision(raw)

            span.set_attribute("react.action", decision.action)
            span.set_attribute("react.model", self.model)
            react_routes_total.add(
                1,
                {"action": decision.action, "model": self.model},
            )
            logger.debug("ReAct 检索路由动作: %s", decision.action)
            return decision

    def _build_prompt(
        self,
        query: str,
        conversation: List[Dict[str, Any]],
        persona: Dict[str, Any],
    ) -> str:
        history = []
        if self.history_messages:
            for message in conversation[-self.history_messages:]:
                role = str(message.get("role", "unknown"))
                content = str(message.get("content", ""))[:500]
                history.append({"role": role, "content": content})

        routing_input = {
            "persona_name": persona.get("name", ""),
            "source_type": persona.get("source_type", "chat"),
            "conversation": history,
            "query": query,
        }
        return (
            "你是 ReAct 对话代理的动作路由器，可使用 retrieval_search 工具。\n"
            "当问题需要人物历史、特定事件、事实依据或上下文记忆时选择 retrieve；"
            "简单问候、闲聊、身份确认或无需外部信息即可回答时选择 respond。\n"
            "只输出 JSON，不要解释，不要输出分析过程。允许的格式只有："
            '{"action":"retrieve"} 或 {"action":"respond"}。\n'
            f"输入：{json.dumps(routing_input, ensure_ascii=False)}"
        )

    def _parse_decision(self, raw: Optional[str]) -> ReActDecision:
        if raw:
            match = _JSON_OBJECT.search(raw)
            if match:
                try:
                    action = str(json.loads(match.group(0)).get("action", "")).lower()
                    if action in {"retrieve", "respond"}:
                        return ReActDecision(action=action)
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass

        logger.warning("ReAct 路由响应无法解析，回退到动作: %s", self.fallback_action)
        return ReActDecision(action=self.fallback_action)
