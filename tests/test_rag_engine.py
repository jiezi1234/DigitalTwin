import pytest
from unittest.mock import MagicMock
from src.rag.rag_engine import RAGEngine
from src.rag.query_processor import QueryProcessor
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
