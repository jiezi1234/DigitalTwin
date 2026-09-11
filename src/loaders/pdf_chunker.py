"""面向教材版面的结构感知 PDF 切分。"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

_HEADING_PATTERN = re.compile(
    r"^(?:第[一二三四五六七八九十百\d]+[章节]|Chapter\s+\d+|"
    r"\d+(?:\.\d+){0,3}\s+\S+)",
    re.IGNORECASE,
)
_CAPTION_PATTERN = re.compile(
    r"^(?:图|表|Figure|Table)\s*[\d一二三四五六七八九十]+(?:[.\-—]\d+)?",
    re.IGNORECASE,
)
_LIST_PATTERN = re.compile(
    r"^(?:[-*•–—]|\d+[.)、]|[（(][一二三四五六七八九十\d]+[）)])\s*"
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])")


@dataclass(frozen=True)
class StructuredPDFChunk:
    content: str
    block_indices: List[int]
    content_types: List[str]
    heading: str
    bbox: Dict[str, float]


class StructureAwarePDFChunker:
    """按标题、段落、列表、图注和表格边界组织教材上下文。"""

    def __init__(
        self,
        target_chars: int = 800,
        max_chars: int = 1200,
        overlap_blocks: int = 1,
    ):
        self.target_chars = max(20, int(target_chars))
        self.max_chars = max(self.target_chars, int(max_chars))
        self.overlap_blocks = max(0, int(overlap_blocks))

    @classmethod
    def classify_block(
        cls,
        text: str,
        is_bold: bool = False,
        font_size: float = 0.0,
        reference_font_size: float = 0.0,
    ) -> str:
        normalized = str(text or "").strip()
        if not normalized:
            return "empty"
        if _CAPTION_PATTERN.match(normalized):
            return "caption"
        visually_prominent = (
            reference_font_size > 0
            and font_size >= reference_font_size * 1.25
            and len(normalized) <= 120
        )
        if (
            _HEADING_PATTERN.match(normalized)
            or visually_prominent
            or (is_bold and len(normalized) <= 80 and "\n" not in normalized)
        ):
            return "heading"
        if _LIST_PATTERN.match(normalized):
            return "list"
        if "\t" in normalized or re.search(r"\S\s{3,}\S", normalized):
            return "table"
        return "paragraph"

    def split_text(self, text: str) -> List[str]:
        """兼容普通 PDFLoader：先按段落/行识别结构，再执行块级切分。"""
        raw_blocks = [
            item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()
        ]
        if len(raw_blocks) == 1:
            raw_blocks = [line.strip() for line in text.splitlines() if line.strip()]
        blocks = [
            {
                "block_index": index,
                "content": content,
                "content_type": self.classify_block(content),
                "bbox": {},
            }
            for index, content in enumerate(raw_blocks)
        ]
        return [chunk.content for chunk in self.chunk_blocks(blocks)]

    def chunk_blocks(
        self, blocks: Sequence[Dict[str, Any]]
    ) -> List[StructuredPDFChunk]:
        chunks: List[StructuredPDFChunk] = []
        current: List[Dict[str, Any]] = []
        current_heading = ""
        previous_was_heading = False

        def flush() -> None:
            nonlocal current
            if not current:
                return
            chunks.append(self._make_chunk(current, current_heading))
            overlap = [
                block
                for block in current
                if block.get("content_type") not in {"heading", "table"}
            ][-self.overlap_blocks :]
            current = list(overlap) if self.overlap_blocks else []

        for raw_block in blocks:
            text = str(raw_block.get("content", "")).strip()
            if not text:
                continue
            block = dict(raw_block)
            block_type = block.get("content_type") or self.classify_block(
                text, bool(block.get("is_bold"))
            )
            block["content_type"] = block_type

            if block_type == "heading":
                flush()
                current = []
                current_heading = (
                    f"{current_heading} {text}" if previous_was_heading else text
                )[:200]
                previous_was_heading = True
                continue

            previous_was_heading = False

            fragments = self._split_oversized_block(block)
            for fragment in fragments:
                if fragment["content_type"] == "table":
                    flush()
                    current = []
                    chunks.append(self._make_chunk([fragment], current_heading))
                    continue

                projected = self._content_length(current, current_heading) + len(
                    fragment["content"]
                )
                if current and projected > self.target_chars:
                    flush()
                current.append(fragment)
                if self._content_length(current, current_heading) >= self.max_chars:
                    flush()

        flush()
        return [chunk for chunk in chunks if chunk.content.strip()]

    def _split_oversized_block(self, block: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = block["content"]
        if len(text) <= self.max_chars:
            return [block]

        sentences = [item for item in _SENTENCE_BOUNDARY.split(text) if item]
        fragments: List[Dict[str, Any]] = []
        buffer = ""
        for sentence in sentences:
            if len(sentence) > self.max_chars:
                if buffer:
                    fragments.append({**block, "content": buffer})
                    buffer = ""
                for start in range(0, len(sentence), self.max_chars):
                    fragments.append(
                        {**block, "content": sentence[start : start + self.max_chars]}
                    )
                continue
            if buffer and len(buffer) + len(sentence) > self.max_chars:
                fragments.append({**block, "content": buffer})
                buffer = sentence
            else:
                buffer += sentence
        if buffer:
            fragments.append({**block, "content": buffer})
        return fragments

    @staticmethod
    def _content_length(blocks: Sequence[Dict[str, Any]], heading: str) -> int:
        return len(heading) + sum(
            len(str(block.get("content", ""))) for block in blocks
        )

    @classmethod
    def _make_chunk(
        cls, blocks: Sequence[Dict[str, Any]], heading: str
    ) -> StructuredPDFChunk:
        body = "\n".join(str(block["content"]).strip() for block in blocks)
        content = f"{heading}\n{body}" if heading and heading not in body else body
        block_indices = [
            int(block["block_index"])
            for block in blocks
            if block.get("block_index") is not None
        ]
        content_types = list(
            dict.fromkeys(
                str(block.get("content_type", "paragraph")) for block in blocks
            )
        )
        return StructuredPDFChunk(
            content=content.strip(),
            block_indices=block_indices,
            content_types=content_types,
            heading=heading,
            bbox=cls._merge_bboxes(blocks),
        )

    @staticmethod
    def _merge_bboxes(blocks: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        boxes = [block.get("bbox") or {} for block in blocks]
        valid = [
            box for box in boxes if all(key in box for key in ("x0", "y0", "x1", "y1"))
        ]
        if not valid:
            return {}
        return {
            "x0": min(float(box["x0"]) for box in valid),
            "y0": min(float(box["y0"]) for box in valid),
            "x1": max(float(box["x1"]) for box in valid),
            "y1": max(float(box["y1"]) for box in valid),
        }
