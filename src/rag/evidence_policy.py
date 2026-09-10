"""检索证据置信度策略。"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

SearchResult = Tuple[str, Dict[str, Any], float]


@dataclass(frozen=True)
class EvidenceAssessment:
    """一次检索结果的证据充分性判断。"""

    sufficient: bool
    best_text_score: float
    best_image_score: float
    supporting_text_count: int
    supporting_image_count: int
    supporting_item_count: int
    min_text_score: float
    min_image_score: float
    min_required_items: int
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceConfidencePolicy:
    """按模态阈值和最少命中数判断检索证据是否足以支撑回答。"""

    TEXT_SCORE_FIELDS = (
        "multimodal_text_score",
        "ocr_text_score",
        "rerank_score",
    )

    def __init__(
        self,
        min_text_score: float = 0.45,
        min_image_score: float = 0.45,
        min_items: int = 1,
    ):
        self.min_text_score = self._normalise_score(min_text_score)
        self.min_image_score = self._normalise_score(min_image_score)
        self.min_items = max(1, int(min_items))

    @staticmethod
    def _normalise_score(value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return min(1.0, max(0.0, score))

    @classmethod
    def _text_score(cls, result: SearchResult) -> float:
        _, metadata, fallback_score = result
        metadata = metadata or {}
        candidates = []
        for field in cls.TEXT_SCORE_FIELDS:
            if field not in metadata:
                continue
            score = cls._normalise_score(metadata[field])
            # 部分模型的 rerank 分数使用 0-100 标度。
            try:
                raw_score = float(metadata[field])
            except (TypeError, ValueError):
                raw_score = 0.0
            if field == "rerank_score" and raw_score > 1:
                score = cls._normalise_score(raw_score / 100)
            candidates.append(score)
        return max(candidates, default=cls._normalise_score(fallback_score))

    @classmethod
    def _image_score(cls, result: Any) -> float:
        if isinstance(result, dict):
            return cls._normalise_score(result.get("score"))
        if isinstance(result, (tuple, list)) and len(result) >= 3:
            return cls._normalise_score(result[2])
        return 0.0

    def assess(
        self,
        text_results: Sequence[SearchResult],
        image_results: Sequence[Any],
    ) -> EvidenceAssessment:
        text_scores = [self._text_score(result) for result in text_results]
        image_scores = [self._image_score(result) for result in image_results]

        supporting_text_count = sum(
            score >= self.min_text_score for score in text_scores
        )
        supporting_image_count = sum(
            score >= self.min_image_score for score in image_scores
        )
        supporting_item_count = supporting_text_count + supporting_image_count

        if not text_scores and not image_scores:
            reason = "no_evidence"
        elif supporting_item_count == 0:
            reason = "below_threshold"
        elif supporting_item_count < self.min_items:
            reason = "insufficient_items"
        else:
            reason = "sufficient"

        return EvidenceAssessment(
            sufficient=reason == "sufficient",
            best_text_score=max(text_scores, default=0.0),
            best_image_score=max(image_scores, default=0.0),
            supporting_text_count=supporting_text_count,
            supporting_image_count=supporting_image_count,
            supporting_item_count=supporting_item_count,
            min_text_score=self.min_text_score,
            min_image_score=self.min_image_score,
            min_required_items=self.min_items,
            reason=reason,
        )
