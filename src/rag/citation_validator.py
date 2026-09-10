"""教材回答引用编号的确定性校验。"""

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

_TEXT_CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class CitationValidation:
    cited_indices: List[int]
    valid_indices: List[int]
    invalid_indices: List[int]
    citation_precision: Optional[float]
    has_valid_citation: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CitationValidator:
    """检查回答中的 ``[n]`` 是否指向实际进入提示词的上下文。"""

    @staticmethod
    def validate(answer: Optional[str], context_count: int) -> CitationValidation:
        cited_indices = []
        seen = set()
        for raw_index in _TEXT_CITATION.findall(str(answer or "")):
            index = int(raw_index)
            if index not in seen:
                cited_indices.append(index)
                seen.add(index)

        valid_indices = [
            index for index in cited_indices if 1 <= index <= max(0, context_count)
        ]
        valid_index_set = set(valid_indices)
        invalid_indices = [
            index for index in cited_indices if index not in valid_index_set
        ]
        citation_precision = (
            len(valid_indices) / len(cited_indices) if cited_indices else None
        )
        return CitationValidation(
            cited_indices=cited_indices,
            valid_indices=valid_indices,
            invalid_indices=invalid_indices,
            citation_precision=citation_precision,
            has_valid_citation=bool(valid_indices),
        )
