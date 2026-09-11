from src.rag.citation_validator import CitationGroundingValidator, CitationValidator


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


def test_grounding_validator_checks_claim_coverage_and_evidence_support():
    validator = CitationGroundingValidator(min_support_score=0.45)
    contexts = [
        "事务具有原子性、一致性、隔离性和持久性。",
        "关系模型使用二维表组织数据。",
    ]

    validation = validator.validate(
        "ACID包括原子性、一致性、隔离性和持久性[1]。关系模型由树组成[2]。还有一句没有引用。",
        contexts,
    )

    assert validation.claim_count == 3
    assert validation.cited_claim_count == 2
    assert validation.supported_claim_count == 1
    assert validation.citation_coverage == 2 / 3
    assert validation.citation_support_precision == 0.5
    assert validation.uncited_claims == ["还有一句没有引用"]
    assert validation.unsupported_claims[0]["citations"] == [2]
