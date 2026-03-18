import os
import pytest
from unittest.mock import patch, MagicMock
from src.infrastructure.llm_client import LLMClient


@pytest.fixture
def mock_env():
    """模拟环境变量"""
    env_vars = {
        "DASHSCOPE_API_KEY": "test-key",
        "LLM_API_BASE": "https://api.example.com",
        "LLM_REWRITING_MODEL": "test-model",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield


def test_llm_client_initialization(mock_env):
    """初始化应该成功"""
    client = LLMClient()
    assert client.api_key == "test-key"
    assert client.model == "test-model"


def test_llm_client_call_success(mock_env):
    """调用 LLM API 应该返回响应"""
    client = LLMClient()

    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "test response"}}]
        }
        mock_post.return_value = mock_response

        result = client.call(
            messages=[{"role": "user", "content": "test"}],
            temperature=0.7,
            max_tokens=100,
        )

        assert result == "test response"


def test_llm_client_call_failure_handling(mock_env):
    """API 调用失败应该返回 None 并记录日志"""
    client = LLMClient()

    with patch("requests.post") as mock_post:
        mock_post.side_effect = Exception("Network error")

        result = client.call(
            messages=[{"role": "user", "content": "test"}],
            temperature=0.7,
            max_tokens=100,
        )

        assert result is None


def test_llm_client_with_telemetry(mock_env):
    """启用 telemetry 时应该创建 span"""
    with patch.dict(os.environ, {"OTEL_ENABLED": "true"}):
        client = LLMClient()

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "test response"}}]
            }
            mock_post.return_value = mock_response

            # 验证调用时能正确处理 tracer
            result = client.call(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.7,
                max_tokens=100,
            )

            assert result == "test response"
