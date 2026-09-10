from src.rag.evidence_policy import EvidenceConfidencePolicy


def test_policy_uses_original_channel_score_instead_of_rrf_score():
    policy = EvidenceConfidencePolicy(min_text_score=0.7)
    results = [
        (
            "文本",
            {"multimodal_text_score": 0.82, "ocr_text_score": 0.76},
            0.032,
        )
    ]

    assessment = policy.assess(results, [])

    assert assessment.sufficient is True
    assert assessment.best_text_score == 0.82
    assert assessment.supporting_text_count == 1


def test_policy_rejects_nonempty_but_low_confidence_results():
    policy = EvidenceConfidencePolicy(min_text_score=0.7, min_image_score=0.7)

    assessment = policy.assess(
        [("相似但无关", {}, 0.42)],
        [{"image_ref": "图1", "score": 0.61}],
    )

    assert assessment.sufficient is False
    assert assessment.reason == "below_threshold"
    assert assessment.supporting_item_count == 0


def test_policy_supports_minimum_evidence_count():
    policy = EvidenceConfidencePolicy(
        min_text_score=0.5, min_image_score=0.5, min_items=2
    )

    assessment = policy.assess([("文本", {}, 0.8)], [])

    assert assessment.sufficient is False
    assert assessment.reason == "insufficient_items"
    assert assessment.min_required_items == 2


def test_policy_accepts_serialized_image_score_and_reports_no_evidence():
    policy = EvidenceConfidencePolicy(min_image_score=0.6)

    image_assessment = policy.assess([], [{"image_ref": "图1", "score": 0.75}])
    empty_assessment = policy.assess([], [])

    assert image_assessment.sufficient is True
    assert image_assessment.best_image_score == 0.75
    assert empty_assessment.reason == "no_evidence"
