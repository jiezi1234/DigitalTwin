import json

import pytest

from src.evaluation.evidence_calibrator import (
    EvidenceCalibrationCase,
    EvidenceThresholdCalibrator,
    load_evidence_calibration_cases,
)


def test_calibrator_selects_boundary_between_positive_and_hard_negative():
    cases = [
        EvidenceCalibrationCase("p1", "text", 0.9, True),
        EvidenceCalibrationCase("p2", "text", 0.8, True),
        EvidenceCalibrationCase("n1", "text", 0.65, False, "hard_negative"),
        EvidenceCalibrationCase("n2", "text", 0.2, False),
    ]

    report = EvidenceThresholdCalibrator().calibrate(cases)
    text_result = report["modalities"]["text"]

    assert text_result["recommended_threshold"] == 0.8
    assert text_result["metrics"]["f1"] == 1.0
    assert text_result["metrics"]["false_positive"] == 0
    assert report["modalities"]["image"] is None


def test_calibrator_prefers_lower_false_positive_rate_when_f1_ties():
    cases = [
        EvidenceCalibrationCase("n", "image", 0.8, False),
    ]

    result = EvidenceThresholdCalibrator().calibrate(cases)["modalities"]["image"]

    assert result["recommended_threshold"] == 1.0
    assert result["metrics"]["false_positive_rate"] == 0.0


def test_load_evidence_calibration_cases_validates_schema(tmp_path):
    dataset = tmp_path / "calibration.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "t1",
                "modality": "text",
                "score": 0.81,
                "sufficient": True,
                "category": "direct_answer",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_evidence_calibration_cases(str(dataset))

    assert cases[0].score == 0.81
    assert cases[0].category == "direct_answer"


def test_load_evidence_calibration_cases_rejects_unknown_modality(tmp_path):
    dataset = tmp_path / "invalid.jsonl"
    dataset.write_text(
        '{"modality":"audio","score":0.8,"sufficient":true}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="modality"):
        load_evidence_calibration_cases(str(dataset))
