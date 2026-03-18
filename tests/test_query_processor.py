import pytest
from unittest.mock import MagicMock
from src.rag.query_processor import QueryProcessor
from src.infrastructure.llm_client import LLMClient


@pytest.fixture
def mock_llm_client():
    """Mock LLM 客户端"""
    client = MagicMock(spec=LLMClient)
    client.call.return_value = "改写后的查询"
    return client


def test_query_processor_initialization(mock_llm_client):
    """初始化查询处理器"""
    processor = QueryProcessor(
        llm_client=mock_llm_client,
        enable_coreference_resolution=True,
        enable_query_rewriting=True,
    )
    assert processor.enable_coreference_resolution is True
    assert processor.enable_query_rewriting is True


def test_query_processor_coreference_resolution(mock_llm_client):
    """测试指代消解"""
    processor = QueryProcessor(
        llm_client=mock_llm_client,
        enable_coreference_resolution=True,
    )

    # 包含代词的查询
    query = "他最近在做什么？"
    resolved = processor.resolve_coreference(query, persona={"name": "张三"})

    # 应该调用 LLM
    mock_llm_client.call.assert_called()


def test_query_processor_rewriting(mock_llm_client):
    """测试查询改写"""
    processor = QueryProcessor(
        llm_client=mock_llm_client,
        enable_query_rewriting=True,
    )

    query = "你怎么样？"
    rewritten = processor.rewrite_query(query, persona={"name": "林黛玉"})

    # 应该调用 LLM
    mock_llm_client.call.assert_called()


def test_query_processor_full_processing(mock_llm_client):
    """测试完整的查询处理流程"""
    processor = QueryProcessor(
        llm_client=mock_llm_client,
        enable_coreference_resolution=True,
        enable_query_rewriting=True,
    )

    query = "他怎么样？"
    result = processor.process(query, persona={"name": "张三"})

    # 应该是处理后的查询
    assert isinstance(result, str)
    assert len(result) > 0
