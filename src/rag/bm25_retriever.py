"""轻量级 BM25 关键词检索器。

索引从 Chroma collection 的原始文本懒加载，不依赖额外分词服务。
中文使用单字与二元词片段，英文和数字按完整词切分。
"""

import logging
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any, Counter as CounterType, Dict, List, Optional, Tuple

from src.infrastructure.db_client import DBClient
from src.infrastructure.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

SearchResult = Tuple[str, Dict[str, Any], float]
_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    """将中英文混合文本转换为适合关键词召回的轻量 token。"""
    tokens: List[str] = []
    for segment in _TOKEN_PATTERN.findall(text or ""):
        if "\u4e00" <= segment[0] <= "\u9fff":
            tokens.extend(segment)
            tokens.extend(
                segment[index : index + 2] for index in range(len(segment) - 1)
            )
        else:
            tokens.append(segment.lower())
    return tokens


@dataclass(frozen=True)
class _BM25Index:
    records: List[SearchResult]
    term_frequencies: List[CounterType[str]]
    document_lengths: List[int]
    document_frequencies: Dict[str, int]
    average_document_length: float


class BM25Retriever:
    """按 collection 懒加载并缓存的 BM25 检索器。"""

    def __init__(self, db_client: DBClient, k1: float = 1.5, b: float = 0.75):
        self.db_client = db_client
        self.k1 = max(0.01, float(k1))
        self.b = min(1.0, max(0.0, float(b)))
        self._indexes: Dict[str, _BM25Index] = {}
        self._lock = threading.RLock()

    def invalidate(self, collection_name: Optional[str] = None) -> None:
        """导入或更新文档后使指定 collection 的缓存失效。"""
        with self._lock:
            if collection_name is None:
                self._indexes.clear()
            else:
                self._indexes.pop(collection_name, None)

    def search(
        self,
        query: str,
        collection_name: str,
        k: int = 15,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """返回按 BM25 分数降序排列的文档。"""
        with tracer.start_as_current_span("bm25.search") as span:
            span.set_attribute("db.collection", collection_name)
            span.set_attribute("bm25.k", k)
            query_tokens = set(tokenize(query))
            if not query_tokens or k <= 0:
                return []

            index, cache_hit = self._get_index(collection_name)
            span.set_attribute("bm25.cache_hit", cache_hit)
            span.set_attribute("bm25.index_size", len(index.records))
            if not index.records:
                return []

            document_count = len(index.records)
            scored_results: List[SearchResult] = []
            for record, frequencies, document_length in zip(
                index.records, index.term_frequencies, index.document_lengths
            ):
                if where and not self._matches_where(record[1], where):
                    continue
                score = 0.0
                length_ratio = document_length / index.average_document_length
                normalization = self.k1 * (1.0 - self.b + self.b * length_ratio)
                for token in query_tokens:
                    term_frequency = frequencies.get(token, 0)
                    if not term_frequency:
                        continue
                    document_frequency = index.document_frequencies[token]
                    inverse_document_frequency = math.log(
                        1.0
                        + (document_count - document_frequency + 0.5)
                        / (document_frequency + 0.5)
                    )
                    score += (
                        inverse_document_frequency
                        * (term_frequency * (self.k1 + 1.0))
                        / (term_frequency + normalization)
                    )

                if score > 0.0:
                    content, metadata, _ = record
                    scored_results.append((content, dict(metadata), score))

            scored_results.sort(key=lambda item: item[2], reverse=True)
            results = scored_results[:k]
            span.set_attribute("bm25.results_count", len(results))
            return results

    @classmethod
    def _matches_where(cls, metadata: Dict[str, Any], where: Dict[str, Any]) -> bool:
        """在缓存索引上执行本项目使用的 Chroma filter 子集。"""
        if "$and" in where:
            return all(cls._matches_where(metadata, item) for item in where["$and"])
        if "$or" in where:
            return any(cls._matches_where(metadata, item) for item in where["$or"])

        for key, condition in where.items():
            actual = metadata.get(key)
            if not isinstance(condition, dict):
                if actual != condition:
                    return False
                continue
            for operator, expected in condition.items():
                try:
                    if operator == "$eq" and actual != expected:
                        return False
                    if operator == "$ne" and actual == expected:
                        return False
                    if operator == "$gt" and not actual > expected:
                        return False
                    if operator == "$gte" and not actual >= expected:
                        return False
                    if operator == "$lt" and not actual < expected:
                        return False
                    if operator == "$lte" and not actual <= expected:
                        return False
                    if operator == "$in" and actual not in expected:
                        return False
                except (TypeError, ValueError):
                    return False
        return True

    def _get_index(self, collection_name: str) -> Tuple[_BM25Index, bool]:
        with self._lock:
            cached = self._indexes.get(collection_name)
            if cached is not None:
                return cached, True

            records = self.db_client.get_records(collection_name=collection_name)
            term_frequencies: List[CounterType[str]] = []
            document_lengths: List[int] = []
            document_frequencies: CounterType[str] = Counter()

            for content, _, _ in records:
                frequencies = Counter(tokenize(content))
                term_frequencies.append(frequencies)
                document_lengths.append(sum(frequencies.values()))
                document_frequencies.update(frequencies.keys())

            average_document_length = (
                sum(document_lengths) / len(document_lengths)
                if document_lengths
                else 1.0
            )
            index = _BM25Index(
                records=records,
                term_frequencies=term_frequencies,
                document_lengths=document_lengths,
                document_frequencies=dict(document_frequencies),
                average_document_length=max(1.0, average_document_length),
            )
            self._indexes[collection_name] = index
            logger.info(
                "BM25 索引已加载: collection=%s, documents=%d",
                collection_name,
                len(records),
            )
            return index, False
