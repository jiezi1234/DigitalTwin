"""教材回答引用编号的确定性校验。"""

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.rag.bm25_retriever import tokenize

_TEXT_CITATION = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])|\n+")


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


@dataclass(frozen=True)
class CitationGroundingValidation:
    claim_count: int
    cited_claim_count: int
    supported_claim_count: int
    citation_coverage: Optional[float]
    citation_support_precision: Optional[float]
    all_claims_cited: bool
    all_cited_claims_supported: bool
    uncited_claims: List[str]
    unsupported_claims: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CitationGroundingValidator:
    """检查事实句是否引用证据，以及引用文本是否与事实句有词汇支撑。"""

    REFUSAL_MARKERS = (
        "没有足够信息",
        "无法从教材",
        "资料中未找到",
        "补充说明（非教材内容）",
    )

    def __init__(self, min_support_score: float = 0.45):
        self.min_support_score = min(1.0, max(0.0, float(min_support_score)))

    def validate(
        self,
        answer: Optional[str],
        contexts: Sequence[Any],
    ) -> CitationGroundingValidation:
        normalized_contexts = [self._context_text(item) for item in contexts]
        claims = self._extract_claims(str(answer or ""))
        cited_claim_count = 0
        supported_claim_count = 0
        uncited_claims = []
        unsupported_claims = []

        for claim in claims:
            citations = list(
                dict.fromkeys(int(item) for item in _TEXT_CITATION.findall(claim))
            )
            if not citations:
                uncited_claims.append(self._clean_claim(claim))
                continue

            cited_claim_count += 1
            cited_contexts = [
                normalized_contexts[index - 1]
                for index in citations
                if 1 <= index <= len(normalized_contexts)
            ]
            support_score = self._support_score(
                self._clean_claim(claim), cited_contexts
            )
            if cited_contexts and support_score >= self.min_support_score:
                supported_claim_count += 1
            else:
                unsupported_claims.append(
                    {
                        "claim": self._clean_claim(claim),
                        "citations": citations,
                        "support_score": round(support_score, 4),
                    }
                )

        claim_count = len(claims)
        citation_coverage = cited_claim_count / claim_count if claim_count else None
        support_precision = (
            supported_claim_count / cited_claim_count if cited_claim_count else None
        )
        return CitationGroundingValidation(
            claim_count=claim_count,
            cited_claim_count=cited_claim_count,
            supported_claim_count=supported_claim_count,
            citation_coverage=citation_coverage,
            citation_support_precision=support_precision,
            all_claims_cited=bool(claim_count) and cited_claim_count == claim_count,
            all_cited_claims_supported=bool(cited_claim_count)
            and supported_claim_count == cited_claim_count,
            uncited_claims=uncited_claims,
            unsupported_claims=unsupported_claims,
        )

    @classmethod
    def _extract_claims(cls, answer: str) -> List[str]:
        claims = []
        for raw_sentence in _SENTENCE_SPLIT.split(answer):
            sentence = raw_sentence.strip().lstrip("#-*• ")
            cleaned = cls._clean_claim(sentence)
            if len(cleaned) < 4:
                continue
            if any(marker in cleaned for marker in cls.REFUSAL_MARKERS):
                continue
            claims.append(sentence)
        return claims

    @staticmethod
    def _clean_claim(claim: str) -> str:
        return _TEXT_CITATION.sub("", claim).strip(" \t。！？!?；;：:")

    @staticmethod
    def _context_text(context: Any) -> str:
        if isinstance(context, (tuple, list)) and context:
            return str(context[0])
        if isinstance(context, dict):
            return str(context.get("content", ""))
        return str(context or "")

    @classmethod
    def _support_score(cls, claim: str, contexts: Sequence[str]) -> float:
        if not claim or not contexts:
            return 0.0
        normalized_claim = re.sub(r"\s+", "", claim).lower()
        normalized_context = re.sub(r"\s+", "", "\n".join(contexts)).lower()
        if normalized_claim and normalized_claim in normalized_context:
            return 1.0

        claim_tokens = cls._meaningful_tokens(claim)
        if not claim_tokens:
            return 0.0
        context_tokens = cls._meaningful_tokens("\n".join(contexts))
        return len(claim_tokens & context_tokens) / len(claim_tokens)

    @staticmethod
    def _meaningful_tokens(text: str) -> set:
        return {token for token in tokenize(text) if len(token) >= 2 or token.isdigit()}
