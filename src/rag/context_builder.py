"""带预算、去重和安全截断的 RAG 上下文构建器。"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.infrastructure.telemetry import get_tracer

tracer = get_tracer(__name__)
SearchResult = Tuple[str, Dict[str, Any], float]


@dataclass(frozen=True)
class ContextBuildResult:
    text: str
    selected_results: List[SearchResult]
    input_count: int
    duplicate_count: int
    truncated_count: int
    used_chars: int

    @property
    def selected_count(self) -> int:
        return len(self.selected_results)


class ContextBuilder:
    """按检索排名构造长度可控、低冗余的模型上下文。"""

    def __init__(
        self,
        max_record_chars: int = 500,
        dedup_threshold: float = 0.90,
        min_remaining_chars: int = 80,
    ):
        self.max_record_chars = max(50, max_record_chars)
        self.dedup_threshold = min(1.0, max(0.0, dedup_threshold))
        self.min_remaining_chars = max(20, min_remaining_chars)

    def build(
        self,
        results: List[SearchResult],
        max_context_length: int = 2000,
        include_metadata: bool = True,
        format_type: str = "chat",
    ) -> ContextBuildResult:
        """去重后按顺序装入上下文预算，并返回构建统计。"""
        max_context_length = max(0, max_context_length)
        if not results or max_context_length == 0:
            return ContextBuildResult("", [], len(results), 0, 0, 0)

        with tracer.start_as_current_span("context.build") as span:
            lines: List[str] = []
            selected_results: List[SearchResult] = []
            selected_signatures: List[str] = []
            duplicate_count = 0
            truncated_count = 0
            used_length = 0

            for content, metadata, score in results:
                signature = self._normalize(content)
                if signature and self._is_duplicate(signature, selected_signatures):
                    duplicate_count += 1
                    continue

                bounded_content = str(content).strip()
                record_truncated = len(bounded_content) > self.max_record_chars
                if record_truncated:
                    bounded_content = self._smart_truncate(
                        bounded_content, self.max_record_chars
                    )

                record = self._format_record(
                    bounded_content,
                    metadata,
                    index=len(selected_results) + 1,
                    include_metadata=include_metadata,
                    format_type=format_type,
                )
                separator_length = 1 if lines else 0
                remaining = max_context_length - used_length - separator_length
                if remaining <= 0:
                    break
                if len(record) > remaining:
                    if remaining < self.min_remaining_chars:
                        break
                    record_overhead = len(record) - len(bounded_content)
                    available_content = remaining - record_overhead
                    if available_content <= 1:
                        break
                    bounded_content = self._smart_truncate(
                        bounded_content, available_content
                    )
                    record = self._format_record(
                        bounded_content,
                        metadata,
                        index=len(selected_results) + 1,
                        include_metadata=include_metadata,
                        format_type=format_type,
                    )
                    record_truncated = True

                lines.append(record)
                selected_results.append((bounded_content, dict(metadata or {}), score))
                selected_signatures.append(signature)
                used_length += separator_length + len(record)
                if record_truncated:
                    truncated_count += 1
                if len(record) >= remaining:
                    break

            text = "\n".join(lines)
            span.set_attribute("context.input_count", len(results))
            span.set_attribute("context.selected_count", len(selected_results))
            span.set_attribute("context.duplicate_count", duplicate_count)
            span.set_attribute("context.truncated_count", truncated_count)
            span.set_attribute("context.used_chars", len(text))
            return ContextBuildResult(
                text=text,
                selected_results=selected_results,
                input_count=len(results),
                duplicate_count=duplicate_count,
                truncated_count=truncated_count,
                used_chars=len(text),
            )

    @staticmethod
    def _normalize(content: str) -> str:
        return re.sub(r"[\W_]+", "", str(content).lower(), flags=re.UNICODE)

    def _is_duplicate(self, candidate: str, selected: List[str]) -> bool:
        for existing in selected:
            if candidate == existing:
                return True
            if min(len(candidate), len(existing)) < 12:
                continue
            candidate_shingles = self._shingles(candidate)
            existing_shingles = self._shingles(existing)
            union = candidate_shingles | existing_shingles
            similarity = (
                len(candidate_shingles & existing_shingles) / len(union)
                if union
                else 0.0
            )
            if similarity >= self.dedup_threshold:
                return True
        return False

    @staticmethod
    def _shingles(text: str, size: int = 3) -> set:
        return {text[index : index + size] for index in range(len(text) - size + 1)}

    @staticmethod
    def _smart_truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        if limit <= 1:
            return text[:limit]
        prefix = text[: limit - 1]
        minimum_boundary = max(0, int(len(prefix) * 0.65))
        boundary = max(
            prefix.rfind(mark) for mark in ("。", "！", "？", ".", "!", "?", "\n")
        )
        if boundary >= minimum_boundary:
            prefix = prefix[: boundary + 1].rstrip()
        return f"{prefix}…"

    @staticmethod
    def _format_record(
        content: str,
        metadata: Dict[str, Any],
        index: int,
        include_metadata: bool,
        format_type: str,
    ) -> str:
        if format_type == "chat":
            if not include_metadata:
                return content
            talker = str(metadata.get("talker", "未知"))
            chat_time = metadata.get("chat_time_str") or metadata.get("chat_time", "")
            time_prefix = f"[{chat_time}] " if chat_time else ""
            already_prefixed = content.startswith(f"{talker}:") or content.startswith(
                f"{talker}@"
            )
            body = content if already_prefixed else f"{talker}: {content}"
            return f"{time_prefix}{body}"

        if format_type == "textbook":
            if not include_metadata:
                return f"[{index}] {content}"
            location_parts = []
            for value in (
                metadata.get("source_file", ""),
                metadata.get("chapter", ""),
                metadata.get("section", ""),
            ):
                if value:
                    location_parts.append(str(value))
            if metadata.get("page", "") not in (None, ""):
                location_parts.append(f"第{metadata['page']}页")
            location = " > ".join(location_parts)
            return f"[{index}]【{location}】\n{content}\n"

        return content
