import pytest
import json
from unittest.mock import MagicMock
from src.rag.query_processor import QueryProcessor, QueryUnderstanding
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
    mock_llm_client.call.assert_called_once()


def test_history_aware_understanding_returns_structured_query(mock_llm_client):
    mock_llm_client.call.return_value = json.dumps(
        {
            "standalone_query": "我在什么时候提到过喜欢杭州",
            "entities": ["杭州"],
            "time_range": None,
        },
        ensure_ascii=False,
    )
    processor = QueryProcessor(
        llm_client=mock_llm_client,
        enable_coreference_resolution=True,
        enable_query_rewriting=True,
        history_messages=4,
    )

    result = processor.understand(
        "那是什么时候？",
        persona={"name": "张三"},
        conversation=[
            {"role": "user", "content": "我以前说过喜欢杭州吗？"},
            {"role": "assistant", "content": "你提到过杭州。"},
        ],
    )

    assert result == QueryUnderstanding(
        original_query="那是什么时候？",
        standalone_query="我在什么时候提到过喜欢杭州",
        entities=["杭州"],
        time_range=None,
    )
    prompt = mock_llm_client.call.call_args.kwargs["messages"][0]["content"]
    assert "我以前说过喜欢杭州吗" in prompt
    assert "那是什么时候" in prompt
    mock_llm_client.call.assert_called_once()


def test_history_is_bounded_and_query_understanding_falls_back(mock_llm_client):
    mock_llm_client.call.return_value = None
    processor = QueryProcessor(
        llm_client=mock_llm_client,
        history_messages=2,
    )
    conversation = [
        {"role": "user", "content": "应该被截断"},
        {"role": "assistant", "content": "最近回答"},
        {"role": "user", "content": "最近问题"},
    ]

    result = processor.process("继续说说", conversation=conversation)

    assert result == "继续说说"
    prompt = mock_llm_client.call.call_args.kwargs["messages"][0]["content"]
    assert "应该被截断" not in prompt
    assert "最近回答" in prompt
    assert "最近问题" in prompt


def test_textbook_query_rewriting_uses_domain_specific_prompt(mock_llm_client):
    processor = QueryProcessor(
        llm_client=mock_llm_client,
        enable_coreference_resolution=False,
        enable_query_rewriting=True,
        domain="textbook",
    )

    processor.rewrite_query("ACID是什么？")

    prompt = mock_llm_client.call.call_args.kwargs["messages"][0]["content"]
    assert "课程教材" in prompt
    assert "分身的聊天历史" not in prompt
