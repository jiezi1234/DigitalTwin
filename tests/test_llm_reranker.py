import json
from unittest.mock import MagicMock

from src.infrastructure.llm_client import LLMClient
from src.rag.llm_reranker import LLMReranker


def test_llm_reranker_orders_candidates_and_preserves_scores():
    llm_client = MagicMock(spec=LLMClient)
    llm_client.call.return_value = json.dumps(
        {
            "scores": [
                {"id": 1, "score": 20},
                {"id": 2, "score": 95},
                {"id": 3, "score": 60},
            ]
        }
    )
    reranker = LLMReranker(llm_client, model="qwen-turbo")
    results = [
        ("候选一", {"source_file": "chat.csv"}, 0.03),
        ("候选二", {"source_file": "chat.csv"}, 0.02),
        ("候选三", {"source_file": "chat.csv"}, 0.01),
    ]

    reranked = reranker.rerank("哪个候选相关", results, top_k=2)

    assert [item[0] for item in reranked] == ["候选二", "候选三"]
    assert reranked[0][1]["rerank_score"] == 95
    assert reranked[0][1]["pre_rerank_rank"] == 2
    assert reranked[0][1]["pre_rerank_score"] == 0.02
    assert reranked[0][2] == 0.95
    assert llm_client.call.call_args.kwargs["model"] == "qwen-turbo"
    assert llm_client.call.call_args.kwargs["temperature"] == 0.0


def test_llm_reranker_falls_back_on_invalid_response():
    llm_client = MagicMock(spec=LLMClient)
    llm_client.call.return_value = "not json"
    results = [("第一", {}, 0.9), ("第二", {}, 0.8)]

    reranked = LLMReranker(llm_client).rerank("查询", results, top_k=1)

    assert reranked == results[:1]


def test_llm_reranker_limits_prompt_candidates_and_appends_unscored_results():
    llm_client = MagicMock(spec=LLMClient)
    llm_client.call.return_value = (
        '{"scores":[{"id":1,"score":10},{"id":2,"score":90}]}'
    )
    results = [
        ("候选一", {}, 0.3),
        ("候选二", {}, 0.2),
        ("不应进入提示词", {}, 0.1),
    ]

    reranked = LLMReranker(llm_client).rerank(
        "查询", results, top_k=3, candidate_limit=2
    )

    assert [item[0] for item in reranked] == ["候选二", "候选一", "不应进入提示词"]
    prompt = llm_client.call.call_args.kwargs["messages"][0]["content"]
    assert "候选二" in prompt
    assert "不应进入提示词" not in prompt


def test_llm_reranker_does_not_call_model_for_one_candidate():
    llm_client = MagicMock(spec=LLMClient)
    results = [("唯一候选", {}, 0.9)]

    assert LLMReranker(llm_client).rerank("查询", results, top_k=1) == results
    llm_client.call.assert_not_called()
