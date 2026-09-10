"""使用标注正例和困难负例校准 RAG 证据阈值。"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List


@dataclass(frozen=True)
class EvidenceCalibrationCase:
    case_id: str
    modality: str
    score: float
    sufficient: bool
    category: str = ""


def load_evidence_calibration_cases(path: str) -> List[EvidenceCalibrationCase]:
    cases = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"证据校准集第 {line_number} 行不是合法 JSON: {exc}"
                ) from exc

            modality = str(payload.get("modality", "")).strip().lower()
            if modality not in {"text", "image"}:
                raise ValueError(
                    f"证据校准集第 {line_number} 行 modality 必须是 text 或 image"
                )
            if not isinstance(payload.get("sufficient"), bool):
                raise ValueError(
                    f"证据校准集第 {line_number} 行 sufficient 必须是布尔值"
                )
            try:
                score = float(payload["score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"证据校准集第 {line_number} 行 score 必须是数字"
                ) from exc
            if not 0 <= score <= 1:
                raise ValueError(f"证据校准集第 {line_number} 行 score 必须位于 0 到 1")

            cases.append(
                EvidenceCalibrationCase(
                    case_id=str(payload.get("id") or f"case-{line_number}"),
                    modality=modality,
                    score=score,
                    sufficient=payload["sufficient"],
                    category=str(payload.get("category", "")),
                )
            )
    if not cases:
        raise ValueError("证据校准集为空")
    return cases


class EvidenceThresholdCalibrator:
    """搜索使 F1 最优的阈值；并列时优先降低误放行率。"""

    def calibrate(self, cases: List[EvidenceCalibrationCase]) -> Dict[str, Any]:
        if not cases:
            raise ValueError("没有可校准证据")

        results = {}
        for modality in ("text", "image"):
            modality_cases = [case for case in cases if case.modality == modality]
            results[modality] = (
                self._calibrate_modality(modality_cases) if modality_cases else None
            )

        return {
            "calibrated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_size": len(cases),
            "selection_rule": "max_f1_then_min_false_positive_rate",
            "modalities": results,
            "cases": [asdict(case) for case in cases],
        }

    def _calibrate_modality(
        self, cases: List[EvidenceCalibrationCase]
    ) -> Dict[str, Any]:
        candidates = sorted({0.0, 1.0, *(case.score for case in cases)})
        evaluated = [self._metrics(cases, threshold) for threshold in candidates]
        best = max(
            evaluated,
            key=lambda item: (
                item["f1"],
                -item["false_positive_rate"],
                item["accuracy"],
                item["threshold"],
            ),
        )
        return {
            "recommended_threshold": best["threshold"],
            "sample_count": len(cases),
            "positive_count": sum(case.sufficient for case in cases),
            "negative_count": sum(not case.sufficient for case in cases),
            "metrics": {
                key: value for key, value in best.items() if key != "threshold"
            },
        }

    @staticmethod
    def _metrics(
        cases: List[EvidenceCalibrationCase], threshold: float
    ) -> Dict[str, Any]:
        tp = fp = tn = fn = 0
        for case in cases:
            predicted = case.score >= threshold
            if predicted and case.sufficient:
                tp += 1
            elif predicted:
                fp += 1
            elif case.sufficient:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        return {
            "threshold": threshold,
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": (tp + tn) / len(cases),
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        }
