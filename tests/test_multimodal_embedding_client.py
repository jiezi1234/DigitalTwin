import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.infrastructure.multimodal_embedding_client import MultiModalEmbeddingClient


@pytest.fixture
def mock_env():
    env_vars = {
        "DASHSCOPE_API_KEY": "test-key",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield


def test_multimodal_client_retries_on_429(mock_env):
    client = MultiModalEmbeddingClient(api_base="https://api.example.com")
    MultiModalEmbeddingClient._retry_not_before = 0.0

    rate_limit_response = MagicMock()
    rate_limit_response.status_code = 429
    rate_limit_response.text = "rate limited"
    rate_limit_response.headers = {"Retry-After": "0.2"}

    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = {
        "output": {"embeddings": [{"index": 0, "embedding": [0.1, 0.2]}]}
    }

    with patch("requests.post", side_effect=[rate_limit_response, success_response]) as mock_post:
        with patch("time.sleep") as mock_sleep:
            result = client.embed_texts(["hello"])

    assert result == [[0.1, 0.2]]
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once()


def test_multimodal_client_does_not_retry_on_400(mock_env):
    client = MultiModalEmbeddingClient(api_base="https://api.example.com")
    MultiModalEmbeddingClient._retry_not_before = 0.0

    bad_request_response = MagicMock()
    bad_request_response.status_code = 400
    bad_request_response.text = "bad request"
    bad_request_response.headers = {}

    with patch("requests.post", return_value=bad_request_response) as mock_post:
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(requests.HTTPError):
                client.embed_texts(["hello"])

    assert mock_post.call_count == 1
    mock_sleep.assert_not_called()


def test_multimodal_client_retries_on_connection_error(mock_env):
    client = MultiModalEmbeddingClient(api_base="https://api.example.com")
    MultiModalEmbeddingClient._retry_not_before = 0.0

    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = {
        "output": {"embeddings": [{"index": 0, "embedding": [0.3, 0.4]}]}
    }

    with patch(
        "requests.post",
        side_effect=[requests.ConnectionError("network"), success_response],
    ) as mock_post:
        with patch("time.sleep") as mock_sleep:
            result = client.embed_texts(["hello"])

    assert result == [[0.3, 0.4]]
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once()


def test_multimodal_client_sets_global_cooldown_on_429(mock_env):
    client = MultiModalEmbeddingClient(api_base="https://api.example.com")
    MultiModalEmbeddingClient._retry_not_before = 0.0

    rate_limit_response = MagicMock()
    rate_limit_response.status_code = 429
    rate_limit_response.text = "rate limited"
    rate_limit_response.headers = {}

    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = {
        "output": {"embeddings": [{"index": 0, "embedding": [0.5, 0.6]}]}
    }

    with patch("requests.post", side_effect=[rate_limit_response, success_response]):
        with patch("time.sleep"):
            result = client.embed_texts(["hello"])

    assert result == [[0.5, 0.6]]
    assert MultiModalEmbeddingClient._retry_not_before > 0.0
