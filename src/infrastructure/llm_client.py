"""
统一的 LLM API 客户端
支持 OpenTelemetry 追踪
"""

import os
import logging
import requests
from typing import List, Dict, Optional
from src.infrastructure.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class LLMClient:
    """统一的大模型 API 调用客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 30,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.api_base = api_base or os.getenv(
            "LLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode"
        )
        self.model = model or os.getenv("LLM_REWRITING_MODEL", "qwen-plus")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set")

    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
        top_p: float = 0.9,
    ) -> Optional[str]:
        """
        调用 LLM API

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大 token 数
            top_p: top_p 采样参数

        Returns:
            生成的文本，失败返回 None
        """
        with tracer.start_as_current_span("llm.api_call") as span:
            try:
                # 设置 span 属性
                span.set_attribute("llm.model", self.model)
                span.set_attribute("llm.temperature", temperature)
                span.set_attribute("llm.max_tokens", max_tokens)

                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "stream": False,
                }

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }

                api_endpoint = f"{self.api_base.rstrip('/')}/v1/chat/completions"

                logger.debug(f"调用 LLM API: {self.model}")
                response = requests.post(
                    api_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("choices") and len(data["choices"]) > 0:
                        result = data["choices"][0].get("message", {}).get("content", "").strip()
                        logger.debug(f"LLM 响应: {result[:100]}...")
                        span.set_attribute("llm.response_status", "success")
                        return result
                else:
                    logger.warning(
                        f"LLM API 错误: {response.status_code} - {response.text[:200]}"
                    )
                    span.set_attribute("llm.response_status", f"error_{response.status_code}")

            except Exception as e:
                logger.warning(f"LLM API 调用失败: {e}")
                span.set_attribute("llm.response_status", "exception")
                span.record_exception(e)

        return None
