import pytest
from unittest.mock import MagicMock, patch
from src.services.rag_service import RAGService
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.db_client import DBClient
from src.rag.bm25_retriever import BM25Retriever
from src.rag.llm_reranker import LLMReranker
from src.rag.react_router import ReActDecision, ReActRetrievalRouter


@pytest.fixture
def mock_components():
    """Mock 所有组件"""
    return {
        "llm_client": MagicMock(spec=LLMClient),
        "db_client": MagicMock(spec=DBClient),
    }


def test_rag_service_initialization(mock_components):
    """初始化 RAG 服务"""
    service = RAGService(
        llm_client=mock_components["llm_client"],
        db_client=mock_components["db_client"],
    )
    assert service.llm_client == mock_components["llm_client"]
    assert service.db_client == mock_components["db_client"]


def test_rag_service_search(mock_components):
    """RAG 搜索"""
    mock_components["db_client"].search.return_value = [
        ("test result", {"source": "chat"}, 0.95),
    ]

    service = RAGService(
        llm_client=mock_components["llm_client"],
        db_client=mock_components["db_client"],
    )

    results = service.search(
        query="test",
        persona={"name": "张三", "doc_count": 100},
    )

    assert len(results) > 0


def test_rag_service_format_context(mock_components):
    """格式化上下文"""
    service = RAGService(
        llm_client=mock_components["llm_client"],
        db_client=mock_components["db_client"],
    )

    results = [("content", {"talker": "张三"}, 0.95)]
    context = service.format_context(results)

    assert "content" in context


def test_rag_service_chat_retrieves_and_generates_once(mock_components):
    """标准 RAG 对话执行一次检索和一次回答生成。"""
    mock_components["db_client"].search.return_value = [
        ("张三: 测试内容", {"talker": "张三"}, 0.95),
    ]
    mock_components["llm_client"].call.return_value = "测试回复"

    service = RAGService(
        llm_client=mock_components["llm_client"],
        db_client=mock_components["db_client"],
        enable_coreference_resolution=False,
        enable_query_rewriting=False,
    )

    reply, retrieval_stats = service.chat(
        query="测试问题",
        conversation=[],
        persona={"name": "张三", "system_prompt": "扮演张三"},
    )

    assert reply == "测试回复"
    assert retrieval_stats == {
        "route_mode": "standard",
        "action": "retrieve",
        "retrieved": True,
        "retrieval_mode": "dense",
        "reranking_enabled": False,
        "reranked": False,
        "result_count": 1,
        "semantic_result_count": 1,
        "neighbor_count": 0,
    }
    mock_components["db_client"].search.assert_called_once()
    mock_components["llm_client"].call.assert_called_once()


def test_rag_service_can_enable_hybrid_search(mock_components):
    mock_components["db_client"].search.return_value = [
        ("向量命中", {"message_index": 1}, 0.9),
    ]
    bm25_retriever = MagicMock(spec=BM25Retriever)
    bm25_retriever.search.return_value = [
        ("关键词命中", {"message_index": 2}, 3.0),
    ]
    service = RAGService(
        llm_client=mock_components["llm_client"],
        db_client=mock_components["db_client"],
        enable_coreference_resolution=False,
        enable_query_rewriting=False,
        enable_neighbor_expansion=False,
        enable_hybrid_search=True,
        hybrid_candidates=12,
        bm25_retriever=bm25_retriever,
    )

    results = service.search(query="杭州", k=2)

    assert {result[0] for result in results} == {"向量命中", "关键词命中"}
    assert mock_components["db_client"].search.call_args.kwargs["k"] == 12
    bm25_retriever.search.assert_called_once()


def test_rag_service_can_enable_candidate_reranking(mock_components):
    initial_results = [
        ("第一候选", {"message_index": 1}, 0.9),
        ("第二候选", {"message_index": 2}, 0.8),
    ]
    reranked_results = [
        ("第二候选", {"message_index": 2, "rerank_score": 95}, 0.95),
    ]
    mock_components["db_client"].search.return_value = initial_results
    reranker = MagicMock(spec=LLMReranker)
    reranker.rerank.return_value = reranked_results
    service = RAGService(
        llm_client=mock_components["llm_client"],
        db_client=mock_components["db_client"],
        enable_coreference_resolution=False,
        enable_query_rewriting=False,
        enable_neighbor_expansion=False,
        enable_reranking=True,
        rerank_candidates=8,
        reranker=reranker,
    )

    results = service.search(query="相关问题", k=1)

    assert results == reranked_results
    assert mock_components["db_client"].search.call_args.kwargs["k"] == 8
    reranker.rerank.assert_called_once_with(
        query="相关问题",
        results=initial_results,
        top_k=1,
        candidate_limit=8,
    )


def test_rag_service_react_can_respond_without_retrieval(mock_components):
    """ReAct 选择直接回答时不调用向量检索。"""
    router = MagicMock(spec=ReActRetrievalRouter)
    router.decide.return_value = ReActDecision(action="respond")
    mock_components["llm_client"].call.return_value = "你好"

    service = RAGService(
        llm_client=mock_components["llm_client"],
        db_client=mock_components["db_client"],
        enable_coreference_resolution=False,
        enable_query_rewriting=False,
        react_router=router,
    )

    reply, retrieval_stats = service.chat(
        query="你好",
        conversation=[],
        persona={"name": "张三", "system_prompt": "扮演张三"},
    )

    assert reply == "你好"
    assert retrieval_stats["route_mode"] == "react"
    assert retrieval_stats["action"] == "respond"
    assert retrieval_stats["retrieved"] is False
    mock_components["db_client"].search.assert_not_called()


def test_rag_service_passes_conversation_to_query_understanding(mock_components):
    mock_components["db_client"].search.return_value = []
    mock_components["llm_client"].call.side_effect = [
        '{"standalone_query":"我什么时候提到杭州","entities":["杭州"],"time_range":null}',
        "测试回复",
    ]
    service = RAGService(
        llm_client=mock_components["llm_client"],
        db_client=mock_components["db_client"],
    )
    conversation = [
        {"role": "user", "content": "我说过喜欢杭州吗？"},
        {"role": "assistant", "content": "说过。"},
    ]

    service.chat(
        query="那是什么时候？",
        conversation=conversation,
        persona={"name": "张三", "system_prompt": "扮演张三"},
    )

    search_call = mock_components["db_client"].search.call_args
    assert search_call.kwargs["query"] == "我什么时候提到杭州"


def test_rag_service_reports_neighbor_expansion(mock_components):
    mock_components["db_client"].search.return_value = [
        ("命中消息", {"conversation_id": "chat-1", "message_index": 2}, 0.9),
    ]
    mock_components["db_client"].get_records.return_value = [
        ("上一条", {"conversation_id": "chat-1", "message_index": 1}, 1.0),
        ("命中消息", {"conversation_id": "chat-1", "message_index": 2}, 1.0),
        ("下一条", {"conversation_id": "chat-1", "message_index": 3}, 1.0),
    ]
    mock_components["llm_client"].call.return_value = "测试回复"
    service = RAGService(
        llm_client=mock_components["llm_client"],
        db_client=mock_components["db_client"],
        enable_coreference_resolution=False,
        enable_query_rewriting=False,
        neighbor_window=1,
        neighbor_anchors=1,
    )

    _, stats = service.chat(
        query="测试问题",
        conversation=[],
        persona={"name": "张三", "system_prompt": "扮演张三"},
    )

    assert stats["semantic_result_count"] == 1
    assert stats["result_count"] == 3
    assert stats["neighbor_count"] == 2
