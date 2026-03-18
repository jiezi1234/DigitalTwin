"""
集成测试：验证从加载、搜索到格式化的完整流程
"""

import pytest
from unittest.mock import MagicMock, patch
from src.loaders.csv_loader import CSVLoader
from src.loaders.base import DataLoaderFactory
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.db_client import DBClient
from src.services.rag_service import RAGService
from src.services.textbook_rag_service import TextbookRAGService
import tempfile
import csv


@pytest.fixture
def sample_csv_file():
    """创建示例 CSV 文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['talker', 'message', 'chat_time'])
        writer.writerow(['张三', '你好', '1609459200'])
        writer.writerow(['李四', '你好啊', '1609459260'])
        writer.writerow(['张三', '最近怎么样', '1609459320'])
        f.flush()
        yield f.name


@pytest.fixture
def mock_llm_client():
    """Mock LLM 客户端"""
    client = MagicMock(spec=LLMClient)
    client.call.return_value = "改写后的查询"
    return client


@pytest.fixture
def mock_db_client():
    """Mock 数据库客户端"""
    client = MagicMock(spec=DBClient)
    return client


def test_end_to_end_chat_rag_flow(mock_llm_client, mock_db_client, sample_csv_file):
    """测试完整的聊天 RAG 流程"""
    # Step 1: 加载数据
    loader = CSVLoader(filepath=sample_csv_file)
    documents = loader.load()
    assert len(documents) == 3

    # Step 2: 初始化 RAG 服务
    rag_service = RAGService(
        llm_client=mock_llm_client,
        db_client=mock_db_client,
    )
    assert rag_service is not None

    # Step 3: 模拟搜索结果
    mock_db_client.search.return_value = [
        (documents[0].content, documents[0].metadata, 0.95),
        (documents[2].content, documents[2].metadata, 0.88),
    ]

    # Step 4: 执行搜索
    persona = {"name": "张三", "doc_count": 100}
    results = rag_service.search(
        query="你最近怎么样",
        persona=persona,
        k=10,
    )
    assert len(results) > 0

    # Step 5: 格式化上下文
    context = rag_service.format_context(results)
    assert "你好" in context
    assert "最近怎么样" in context


def test_end_to_end_textbook_rag_flow(mock_llm_client, mock_db_client):
    """测试完整的教材 RAG 流程"""
    # 初始化教材 RAG 服务
    textbook_service = TextbookRAGService(
        llm_client=mock_llm_client,
        db_client=mock_db_client,
    )
    assert textbook_service is not None

    # 模拟搜索结果
    mock_db_client.search.return_value = [
        ("第一章讲述了基础概念", {"source": "pdf", "page": 1, "chapter": "第一章"}, 0.92),
        ("第二章详细讨论了高级主题", {"source": "pdf", "page": 15, "chapter": "第二章"}, 0.85),
    ]

    # 执行搜索
    results = textbook_service.search(query="什么是基础概念", k=10)
    assert len(results) > 0

    # 格式化上下文
    context = textbook_service.format_context(results)
    assert "基础概念" in context or "高级主题" in context


def test_loader_factory_integration(sample_csv_file):
    """测试加载器工厂集成"""
    DataLoaderFactory.register("csv", CSVLoader)

    # 使用工厂创建加载器
    loader = DataLoaderFactory.create("csv", filepath=sample_csv_file)
    assert loader is not None

    # 加载数据
    documents = loader.load()
    assert len(documents) == 3
    assert all(hasattr(doc, 'content') for doc in documents)
    assert all(hasattr(doc, 'metadata') for doc in documents)


def test_query_processing_integration(mock_llm_client):
    """测试查询处理集成"""
    from src.rag.query_processor import QueryProcessor

    processor = QueryProcessor(
        llm_client=mock_llm_client,
        enable_coreference_resolution=True,
        enable_query_rewriting=True,
    )

    # 测试带代词的查询
    query = "他最近怎么样？"
    persona = {"name": "张三", "doc_count": 100}
    result = processor.process(query, persona=persona)

    # 应该调用了 LLM
    assert mock_llm_client.call.called
    assert isinstance(result, str)


def test_rag_engine_integration(mock_db_client):
    """测试 RAG 引擎集成"""
    from src.rag.rag_engine import RAGEngine

    engine = RAGEngine(db_client=mock_db_client)

    # 模拟搜索结果
    mock_db_client.search.return_value = [
        ("内容1", {"talker": "张三", "source": "chat"}, 0.95),
        ("内容2", {"talker": "李四", "source": "chat"}, 0.88),
    ]

    # 测试搜索
    results = engine.search(
        query="test",
        collection_name="test_collection",
        k=10,
    )
    assert len(results) == 2

    # 测试格式化（聊天模式）
    context_chat = engine.format_context(results, format_type="chat")
    assert "内容1" in context_chat

    # 测试格式化（教材模式）
    context_textbook = engine.format_context(results, format_type="textbook")
    assert "内容1" in context_textbook or "内容2" in context_textbook
