import json

import pytest

from src.evaluation.retrieval_evaluator import (
    EvaluationCase,
    RetrievalEvaluator,
    load_evaluation_cases,
)


def test_load_evaluation_cases(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "q1",
                "query": "测试问题",
                "relevant": [{"source_file": "book.pdf", "page": 2}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_evaluation_cases(str(dataset))

    assert cases[0].case_id == "q1"
    assert cases[0].relevant[0]["page"] == 2


def test_load_evaluation_cases_rejects_missing_labels(tmp_path):
    dataset = tmp_path / "invalid.jsonl"
    dataset.write_text('{"query":"测试"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="relevant"):
        load_evaluation_cases(str(dataset))


def test_retrieval_metrics_are_computed_at_k():
    cases = [
        EvaluationCase(
            "q1", "问题1", [{"source_file": "a.csv", "content_contains": "目标"}]
        ),
        EvaluationCase("q2", "问题2", [{"source_file": "b.csv", "page": 3}]),
    ]

    def retriever(case, _k):
        if case.case_id == "q1":
            return [
                ("无关内容", {"source_file": "x.csv"}, 0.9),
                ("这是目标内容", {"source_file": "a.csv"}, 0.8),
            ]
        return [("未命中", {"source_file": "b.csv", "page": 1}, 0.7)]

    report = RetrievalEvaluator(k=5).evaluate(cases, retriever, "test")

    assert report["metrics"]["hit_rate@5"] == 0.5
    assert report["metrics"]["mrr@5"] == 0.25
    assert report["metrics"]["recall@5"] == 0.5
    assert report["metrics"]["context_precision@5"] == 0.25


def test_compare_reports_improved_and_regressed_cases():
    cases = [
        EvaluationCase("better", "问题1", [{"content_contains": "目标"}]),
        EvaluationCase("worse", "问题2", [{"content_contains": "目标"}]),
    ]

    def baseline(case, _k):
        text = "目标" if case.case_id == "worse" else "无关"
        return [(text, {}, 1.0)]

    def optimized(case, _k):
        text = "目标" if case.case_id == "better" else "无关"
        return [(text, {}, 1.0)]

    report = RetrievalEvaluator(k=5).compare(cases, baseline, optimized)

    assert report["delta"]["improved_cases"] == ["better"]
    assert report["delta"]["regressed_cases"] == ["worse"]
