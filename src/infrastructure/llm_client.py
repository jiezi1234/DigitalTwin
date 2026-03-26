"""
统一的 LLM API 客户端
支持 OpenTelemetry 追踪
"""

import os
import logging
import time
import requests
from typing import List, Dict, Optional
from src.infrastructure.telemetry import get_tracer, get_meter

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)
meter = get_meter(__name__)

# 定义指标
llm_calls_total = meter.create_counter(
    "llm_calls_total",
    description="Total number of LLM API calls",
    unit="1"
)
llm_call_duration = meter.create_histogram(
    "llm_call_duration_seconds",
    description="Duration of LLM API calls",
    unit="s"
)


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
        model: Optional[str] = None,
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
        start_time = time.time()
        active_model = model or self.model
        attributes = {"llm.model": active_model}
        status = "unknown" # Default status

        with tracer.start_as_current_span("llm.api_call") as span:
            try:
                # 设置 span 属性
                span.set_attribute("llm.model", active_model)
                span.set_attribute("llm.temperature", temperature)
                span.set_attribute("llm.max_tokens", max_tokens)

                payload = {
                    "model": active_model,
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

                logger.debug(f"调用 LLM API: {active_model}")
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
                        status = "success"
                        return result
                else:
                    logger.warning(
                        f"LLM API 错误: {response.status_code} - {response.text[:200]}"
                    )
                    span.set_attribute("llm.response_status", f"error_{response.status_code}")
                    status = f"error_{response.status_code}"

            except Exception as e:
                logger.warning(f"LLM API 调用失败: {e}")
                span.set_attribute("llm.response_status", "exception")
                span.record_exception(e)
                status = "exception"
            finally:
                duration = time.time() - start_time
                attributes["status"] = status
                llm_calls_total.add(1, attributes)
                llm_call_duration.record(duration, attributes)

        return None

    def call_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 0.9,
        model: Optional[str] = None,
    ):
        """
        调用 LLM API 并返回生成器实现流式输出

        Args:
            messages: 消息列表
            temperature: 温度
            max_tokens: 最大长度
            top_p: top_p

        Yields:
            生成的文本片段
        """
        start_time = time.time()
        active_model = model or self.model
        attributes = {"llm.model": active_model, "stream": "true"}
        status = "unknown"

        with tracer.start_as_current_span("llm.api_call_stream") as span:
            try:
                payload = {
                    "model": active_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "stream": True,
                }

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }

                api_endpoint = f"{self.api_base.rstrip('/')}/v1/chat/completions"

                # 流式无法简单衡量单次“耗时”，但可以记录调用开始
                llm_calls_total.add(1, attributes)
                
                response = requests.post(
                    api_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    stream=True
                )

                if response.status_code == 200:
                    import json
                    status = "success"
                    for line in response.iter_lines():
                        if not line:
                            continue
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            data_content = line_str[6:].strip()
                            if data_content == "[DONE]":
                                break
                            try:
                                data = json.loads(data_content)
                                if data.get("choices") and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                            except Exception:
                                continue
                else:
                    logger.warning(f"LLM API Stream 错误: {response.status_code}")
                    status = f"error_{response.status_code}"
            except Exception as e:
                logger.warning(f"LLM API Stream 调用失败: {e}")
                span.record_exception(e)
                status = "exception"
            finally:
                # 记录首包耗时或建立连接耗时，这里简单记录总耗时
                duration = time.time() - start_time
                attributes["status"] = status
                llm_call_duration.record(duration, attributes)
