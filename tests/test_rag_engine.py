import pytest
from unittest.mock import MagicMock
from src.rag.rag_engine import RAGEngine
from src.rag.query_processor import QueryProcessor
from src.rag.bm25_retriever import BM25Retriever
from src.rag.llm_reranker import LLMReranker
from src.infrastructure.db_client import DBClient


@pytest.fixture
def mock_db_client():
    """Mock DB 客户端"""
    client = MagicMock(spec=DBClient)
    client.search.return_value = [
        ("test content 1", {"source": "chat"}, 0.95),
        ("test content 2", {"source": "chat"}, 0.88),
    ]
    return client


@pytest.fixture
def mock_query_processor():
    """Mock 查询处理器"""
    processor = MagicMock(spec=QueryProcessor)
    processor.process.return_value = "处理后的查询"
    return processor


def test_rag_engine_initialization(mock_db_client):
    """初始化 RAG 引擎"""
    engine = RAGEngine(db_client=mock_db_client)
    assert engine.db_client == mock_db_client


def test_rag_engine_search(mock_db_client, mock_query_processor):
    """搜索向量数据库"""
    engine = RAGEngine(db_client=mock_db_client)

    results = engine.search(
        query="test query",
        collection_name="test_collection",
        query_processor=mock_query_processor,
        k=10,
    )

    assert len(results) == 2
    assert results[0][0] == "test content 1"


def test_rag_engine_fuses_dense_and_bm25_with_weighted_rrf(mock_db_client):
    mock_db_client.search.return_value = [
        ("dense only", {"message_index": 1}, 0.9),
        ("both", {"message_index": 2}, 0.8),
    ]
    bm25_retriever = MagicMock(spec=BM25Retriever)
    bm25_retriever.search.return_value = [
        ("both", {"message_index": 2}, 4.2),
        ("keyword only", {"message_index": 3}, 3.1),
    ]
    engine = RAGEngine(
        db_client=mock_db_client,
        lexical_retriever=bm25_retriever,
    )

    results = engine.search(
        query="test",
        collection_name="persona",
        k=3,
        hybrid_search=True,
        hybrid_candidates=20,
    )

    assert results[0][0] == "both"
    assert results[0][1]["retrieval_channels"] == "dense,bm25"
    assert results[0][1]["dense_score"] == 0.8
    assert results[0][1]["bm25_score"] == 4.2
    assert mock_db_client.search.call_args.kwargs["k"] == 20
    bm25_retriever.search.assert_called_once_with(
        query="test",
        collection_name="persona",
        k=20,
    )


def test_rag_engine_falls_back_to_dense_when_bm25_fails(mock_db_client):
    bm25_retriever = MagicMock(spec=BM25Retriever)
    bm25_retriever.search.side_effect = RuntimeError("BM25 unavailable")
    engine = RAGEngine(
        db_client=mock_db_client,
        lexical_retriever=bm25_retriever,
    )

    results = engine.search(
        query="test",
        collection_name="persona",
        hybrid_search=True,
    )

    assert [result[0] for result in results] == [
        "test content 1",
        "test content 2",
    ]
    assert results[0][1]["retrieval_channels"] == "dense"


def test_rag_engine_falls_back_to_bm25_when_dense_fails(mock_db_client):
    mock_db_client.search.side_effect = RuntimeError("Dense unavailable")
    bm25_retriever = MagicMock(spec=BM25Retriever)
    bm25_retriever.search.return_value = [("keyword", {}, 2.0)]
    engine = RAGEngine(
        db_client=mock_db_client,
        lexical_retriever=bm25_retriever,
    )

    results = engine.search(
        query="test",
        collection_name="persona",
        hybrid_search=True,
    )

    assert results[0][0] == "keyword"
    assert results[0][1]["retrieval_channels"] == "bm25"


def test_rag_engine_retrieves_extra_candidates_before_reranking(
    mock_db_client, mock_query_processor
):
    reranker = MagicMock(spec=LLMReranker)
    reranker.rerank.return_value = [("reranked", {"rerank_score": 90}, 0.9)]
    engine = RAGEngine(db_client=mock_db_client, reranker=reranker)

    results = engine.search(
        query="test",
        collection_name="persona",
        query_processor=mock_query_processor,
        k=1,
        rerank=True,
        rerank_candidates=10,
    )

    assert results[0][0] == "reranked"
    assert mock_db_client.search.call_args.kwargs["k"] == 10
    assert mock_db_client.search.call_args.kwargs["query"] == "处理后的查询"
    reranker.rerank.assert_called_once_with(
        query="处理后的查询",
        results=mock_db_client.search.return_value,
        top_k=1,
        candidate_limit=10,
    )


def test_rag_engine_format_context(mock_db_client):
    """格式化搜索结果为上下文"""
    engine = RAGEngine(db_client=mock_db_client)

    results = [
        ("content 1", {"source": "chat", "talker": "张三"}, 0.95),
        ("content 2", {"source": "chat", "talker": "李四"}, 0.88),
    ]

    context = engine.format_context(results, max_context_length=1000)

    assert "content 1" in context
    assert "content 2" in context


def test_rag_engine_expands_and_deduplicates_chat_neighbors(mock_db_client):
    engine = RAGEngine(db_client=mock_db_client)
    semantic_results = [
        ("命中消息", {"conversation_id": "chat-1", "message_index": 2}, 0.9),
        ("下一条", {"conversation_id": "chat-1", "message_index": 3}, 0.8),
    ]
    mock_db_client.get_records.side_effect = [
        [
            ("上一条", {"conversation_id": "chat-1", "message_index": 1}, 1.0),
            ("命中消息", {"conversation_id": "chat-1", "message_index": 2}, 1.0),
            ("下一条", {"conversation_id": "chat-1", "message_index": 3}, 1.0),
        ],
        [
            ("命中消息", {"conversation_id": "chat-1", "message_index": 2}, 1.0),
            ("下一条", {"conversation_id": "chat-1", "message_index": 3}, 1.0),
            ("后续消息", {"conversation_id": "chat-1", "message_index": 4}, 1.0),
        ],
    ]

    expanded = engine.expand_chat_neighbors(
        semantic_results,
        collection_name="persona",
        window_size=1,
        anchor_limit=2,
        max_results=10,
    )

    assert [item[0] for item in expanded] == [
        "上一条",
        "命中消息",
        "下一条",
        "后续消息",
    ]
    assert expanded[0][1]["retrieval_origin"] == "neighbor"
    assert expanded[1][1]["retrieval_origin"] == "semantic"
    assert expanded[1][1]["neighbor_distance"] == 0
    where = mock_db_client.get_records.call_args_list[0].kwargs["where"]
    assert {"message_index": {"$gte": 1}} in where["$and"]
    assert {"message_index": {"$lte": 3}} in where["$and"]


def test_rag_engine_keeps_legacy_results_without_neighbor_metadata(mock_db_client):
    engine = RAGEngine(db_client=mock_db_client)
    results = [("旧消息", {"source_file": "chat.csv", "chat_time": "1"}, 0.9)]

    expanded = engine.expand_chat_neighbors(results, collection_name="persona")

    assert expanded[0][0] == "旧消息"
    assert expanded[0][1]["retrieval_origin"] == "semantic"
    mock_db_client.get_records.assert_not_called()
