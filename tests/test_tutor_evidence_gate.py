from unittest.mock import MagicMock

from src.api.app import create_app
from src.api.routes import tutor
from src.rag.citation_validator import CitationGroundingValidator, CitationValidator
from src.rag.context_builder import ContextBuildResult
from src.rag.evidence_policy import EvidenceConfidencePolicy


def test_tutor_rejects_low_confidence_hits_without_calling_llm(monkeypatch):
    low_score_result = ("主题相似但不能回答", {"source_file": "book.pdf"}, 0.2)
    service = MagicMock()
    service.retrieve.return_value = {
        "text_results": [low_score_result],
        "ocr_text_results": [],
        "image_results": [],
    }
    service.build_context.return_value = ContextBuildResult(
        text="[1] 主题相似但不能回答",
        selected_results=[low_score_result],
        input_count=1,
        duplicate_count=0,
        truncated_count=0,
        used_chars=14,
    )
    assessment = EvidenceConfidencePolicy(min_text_score=0.7).assess(
        [low_score_result], []
    )
    service.assess_evidence.return_value = assessment
    service.validate_citations.side_effect = (
        lambda reply, results: CitationValidator.validate(reply, len(results))
    )
    service.validate_citation_grounding.side_effect = (
        lambda reply, results: CitationGroundingValidator().validate(reply, results)
    )

    monkeypatch.setattr(tutor, "get_tutor_service", lambda: service)
    llm_call = MagicMock()
    monkeypatch.setattr(tutor.llm_client, "call", llm_call)
    tutor.tutor_sessions.clear()
    app = create_app()

    response = app.test_client().post(
        "/tutor/chat",
        json={"message": "这个概念是什么？", "stream": False},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["quality"]["evidence_sufficient"] is False
    assert payload["quality"]["evidence"]["reason"] == "below_threshold"
    assert payload["sources"] == []
    llm_call.assert_not_called()
