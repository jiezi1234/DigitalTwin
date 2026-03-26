"""
多模态 Embedding 客户端
直接调用 DashScope multimodal-embedding-v1 HTTP API。
"""

import base64
import mimetypes
import os
import time
import random
import logging
import threading
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class MultiModalEmbeddingClient:
    """DashScope 多模态向量客户端"""

    _gate_lock = threading.Lock()
    _request_gate: Optional[threading.BoundedSemaphore] = None
    _request_gate_size: Optional[int] = None
    _retry_not_before: float = 0.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = "multimodal-embedding-v1",
        timeout: int = 60,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.api_base = api_base or os.getenv(
            "DASHSCOPE_MM_API_BASE",
            "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
        )
        self.model = model
        self.timeout = timeout
        self.max_concurrency = int(os.getenv("DASHSCOPE_MM_MAX_CONCURRENCY", "2"))

        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set")

        with self._gate_lock:
            if (
                self.__class__._request_gate is None
                or self.__class__._request_gate_size != self.max_concurrency
            ):
                self.__class__._request_gate = threading.BoundedSemaphore(self.max_concurrency)
                self.__class__._request_gate_size = self.max_concurrency

    @staticmethod
    def _image_to_data_uri(image_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/png"

        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _request_embeddings(
        self,
        contents: List[Dict[str, Any]],
        instruct: Optional[str] = None,
        max_retries: int = 5,
    ) -> List[List[float]]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": {"contents": contents},
        }
        if instruct:
            payload["parameters"] = {"instruct": instruct}

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
                embeddings = data.get("output", {}).get("embeddings", [])
                embeddings = sorted(embeddings, key=lambda item: item.get("index", 0))
                return [item["embedding"] for item in embeddings]
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise
                if not self._is_retryable_error(exc):
                    raise
                wait_time = self._compute_retry_delay(exc, attempt)
                cooldown_applied = self._apply_global_cooldown(exc, wait_time)
                logger.warning(
                    "Multimodal embedding 请求失败 (attempt=%s/%s): %s; %.2fs 后重试",
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
        jitter = random.uniform(0.1, 0.8)
        return base_delay + jitter

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
    def embed_texts(
        self,
        texts: List[str],
        instruct: str = "Represent the query or document for textbook retrieval.",
    ) -> List[List[float]]:
        contents = [{"text": text} for text in texts]
        return self._request_embeddings(contents, instruct=instruct)

    def embed_query(self, query: str) -> List[float]:
        results = self.embed_texts(
            [query],
            instruct="Represent the query for retrieving textbook text blocks and images.",
        )
        return results[0]

    def embed_images(
        self,
        image_paths: List[str],
        instruct: str = "Represent the image for textbook retrieval.",
    ) -> List[List[float]]:
        contents = [{"image": self._image_to_data_uri(path)} for path in image_paths]
        return self._request_embeddings(contents, instruct=instruct)
