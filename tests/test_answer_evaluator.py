import json
from unittest.mock import MagicMock

import pytest

from src.evaluation.answer_evaluator import (
    AnswerEvaluationCase,
    AnswerEvaluator,
    LLMGroundednessJudge,
    load_answer_cases,
)
from src.infrastructure.llm_client import LLMClient


def test_load_answer_cases_supports_string_and_object_contexts(tmp_path):
    dataset = tmp_path / "answers.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "a1",
                "query": "ACID是什么",
                "contexts": ["上下文一", {"content": "上下文二", "page": 2}],
                "answer": "回答[1]",
                "answerable": True,
                "expected_contains": ["回答"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_answer_cases(str(dataset))

    assert cases[0].contexts == ["上下文一", "上下文二"]
    assert cases[0].answerable is True


def test_load_answer_cases_rejects_missing_answerable(tmp_path):
    dataset = tmp_path / "invalid.jsonl"
    dataset.write_text(
        '{"query":"问题","contexts":[],"answer":"回答"}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="answerable"):
        load_answer_cases(str(dataset))


def test_answer_evaluator_computes_citation_abstention_and_coverage_metrics():
    cases = [
        AnswerEvaluationCase(
            case_id="answerable",
            query="ACID是什么",
            answer="包括原子性和一致性[1]。",
            contexts=["事务具有原子性和一致性"],
            answerable=True,
            expected_contains=["原子性", "一致性"],
        ),
        AnswerEvaluationCase(
            case_id="unanswerable",
            query="作者爱好是什么",
            answer="当前教材资料中没有足够信息回答这个问题。",
            contexts=[],
            answerable=False,
        ),
    ]

    report = AnswerEvaluator().evaluate(cases)

    assert report["metrics"] == {
        "abstention_accuracy": 1.0,
        "citation_precision": 1.0,
        "citation_coverage": 1.0,
        "answer_keyword_recall": 1.0,
        "groundedness": None,
    }


def test_llm_groundedness_judge_parses_and_clamps_score():
    llm_client = MagicMock(spec=LLMClient)
    llm_client.call.return_value = '{"groundedness":1.2,"reason":"有依据"}'
    case = AnswerEvaluationCase("a", "q", "回答", ["证据"], True)

    score = LLMGroundednessJudge(llm_client, model="judge-model")(case)

    assert score == 1.0
    assert llm_client.call.call_args.kwargs["model"] == "judge-model"


def test_answer_evaluator_uses_optional_groundedness_judge():
    judge = MagicMock(return_value=0.75)
    case = AnswerEvaluationCase("a", "q", "回答[1]", ["证据"], True)

    report = AnswerEvaluator(groundedness_judge=judge).evaluate([case])

    assert report["metrics"]["groundedness"] == 0.75
    judge.assert_called_once_with(case)
