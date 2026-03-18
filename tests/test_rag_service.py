import pytest
from unittest.mock import MagicMock, patch
from src.services.rag_service import RAGService
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.db_client import DBClient


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
