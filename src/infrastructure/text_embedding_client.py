"""
文本 Embedding 客户端
直接调用 DashScope OpenAI-compatible embeddings API。
"""

import logging
import os
import random
import threading
import time
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)


class TextEmbeddingClient:
    """DashScope 文本向量客户端"""

    _gate_lock = threading.Lock()
    _request_gate: Optional[threading.BoundedSemaphore] = None
    _request_gate_size: Optional[int] = None
    _retry_not_before: float = 0.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = "text-embedding-v4",
        timeout: int = 60,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.api_base = api_base or os.getenv(
            "DASHSCOPE_TEXT_EMBED_API_BASE",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
        )
        self.model = model
        self.timeout = timeout
        self.max_concurrency = int(os.getenv("DASHSCOPE_TEXT_MAX_CONCURRENCY", "2"))

        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set")

        with self._gate_lock:
            if (
                self.__class__._request_gate is None
                or self.__class__._request_gate_size != self.max_concurrency
            ):
                self.__class__._request_gate = threading.BoundedSemaphore(self.max_concurrency)
                self.__class__._request_gate_size = self.max_concurrency

    def embed_texts(self, texts: List[str], max_retries: int = 5) -> List[List[float]]:
        payload = {
            "model": self.model,
            "input": texts,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(max_retries):
            try:
                self._wait_for_global_cooldown()
                with self.__class__._request_gate:
                    response = requests.post(
                        self.api_base,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout,
                    )
                if response.status_code >= 400:
                    error = requests.HTTPError(
                        f"{response.status_code} Client Error: {response.text[:500]}",
                        response=response,
                    )
                    if not self._is_retryable_http_status(response.status_code):
                        raise error
                    raise error

                data = response.json()
                vectors = data.get("data", [])
                vectors = sorted(vectors, key=lambda item: item.get("index", 0))
                return [item["embedding"] for item in vectors]
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise
                if not self._is_retryable_error(exc):
                    raise
                wait_time = self._compute_retry_delay(exc, attempt)
                cooldown_applied = self._apply_global_cooldown(exc, wait_time)
                logger.warning(
                    "Text embedding 请求失败 (attempt=%s/%s): %s; %.2fs 后重试",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait_time,
                )
                if not cooldown_applied:
                    time.sleep(wait_time)

        return []

    @staticmethod
    def _is_retryable_http_status(status_code: int) -> bool:
        return status_code == 429 or 500 <= status_code < 600

    @classmethod
    def _is_retryable_error(cls, exc: Exception) -> bool:
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(exc, requests.HTTPError):
            response = getattr(exc, "response", None)
            if response is None:
                return False
            return cls._is_retryable_http_status(response.status_code)
        return False

    @staticmethod
    def _compute_retry_delay(exc: Exception, attempt: int) -> float:
        response = getattr(exc, "response", None)
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(0.5, float(retry_after))
                except ValueError:
                    pass
        if response is not None and response.status_code == 429:
            base_delay = min(60.0, 5.0 * (2 ** attempt))
        else:
            base_delay = min(30.0, 2 ** attempt)
        return base_delay + random.uniform(0.1, 0.8)

    @classmethod
    def _wait_for_global_cooldown(cls) -> None:
        with cls._gate_lock:
            wait_until = cls._retry_not_before
        now = time.time()
        if now >= wait_until:
            return
        time.sleep(wait_until - now)

    @classmethod
    def _apply_global_cooldown(cls, exc: Exception, wait_time: float) -> bool:
        response = getattr(exc, "response", None)
        if isinstance(exc, requests.HTTPError) and response is not None and response.status_code == 429:
            with cls._gate_lock:
                cls._retry_not_before = max(cls._retry_not_before, time.time() + wait_time)
            return True
        return False
