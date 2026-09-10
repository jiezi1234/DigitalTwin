from unittest.mock import MagicMock

from src.infrastructure.db_client import DBClient
from src.rag.bm25_retriever import BM25Retriever, tokenize


def test_tokenize_supports_chinese_bigrams_and_english_words():
    tokens = tokenize("杭州 RAG2026")

    assert "杭" in tokens
    assert "杭州" in tokens
    assert "rag2026" in tokens


def test_bm25_retriever_ranks_keyword_match_first_and_reuses_cache():
    db_client = MagicMock(spec=DBClient)
    db_client.get_records.return_value = [
        ("我喜欢去杭州旅行", {"message_index": 1}, 1.0),
        ("数据库事务具有原子性", {"message_index": 2}, 1.0),
        ("周末打羽毛球", {"message_index": 3}, 1.0),
    ]
    retriever = BM25Retriever(db_client)

    first_results = retriever.search("杭州", "persona", k=3)
    second_results = retriever.search("喜欢杭州", "persona", k=3)

    assert first_results[0][0] == "我喜欢去杭州旅行"
    assert second_results[0][0] == "我喜欢去杭州旅行"
    db_client.get_records.assert_called_once_with(collection_name="persona")


def test_bm25_retriever_invalidation_reloads_collection():
    db_client = MagicMock(spec=DBClient)
    db_client.get_records.return_value = [("杭州", {}, 1.0)]
    retriever = BM25Retriever(db_client)

    retriever.search("杭州", "persona")
    retriever.invalidate("persona")
    retriever.search("杭州", "persona")

    assert db_client.get_records.call_count == 2


def test_bm25_retriever_returns_empty_for_unmatched_query():
    db_client = MagicMock(spec=DBClient)
    db_client.get_records.return_value = [("数据库事务", {}, 1.0)]

    assert BM25Retriever(db_client).search("杭州", "persona") == []


def test_bm25_retriever_applies_metadata_filter_before_ranking():
    db_client = MagicMock(spec=DBClient)
    db_client.get_records.return_value = [
        ("杭州旅行", {"chat_time": 100}, 1.0),
        ("杭州工作", {"chat_time": 200}, 1.0),
    ]
    retriever = BM25Retriever(db_client)

    results = retriever.search(
        "杭州",
        "persona",
        where={
            "$and": [
                {"chat_time": {"$gte": 150}},
                {"chat_time": {"$lte": 250}},
            ]
        },
    )

    assert [item[0] for item in results] == ["杭州工作"]
