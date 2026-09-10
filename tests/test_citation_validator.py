from src.rag.citation_validator import CitationValidator


def test_citation_validator_reports_valid_and_invalid_indices():
    validation = CitationValidator.validate(
        "原子性[1]，图示见[图1]，错误来源[4]，再次引用[1]。",
        context_count=2,
    )

    assert validation.cited_indices == [1, 4]
    assert validation.valid_indices == [1]
    assert validation.invalid_indices == [4]
    assert validation.citation_precision == 0.5
    assert validation.has_valid_citation is True


def test_citation_validator_distinguishes_no_citation():
    validation = CitationValidator.validate("没有引用", context_count=3)

    assert validation.cited_indices == []
    assert validation.citation_precision is None
    assert validation.has_valid_citation is False
