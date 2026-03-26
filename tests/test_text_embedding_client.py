import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.infrastructure.text_embedding_client import TextEmbeddingClient


@pytest.fixture
def mock_env():
    env_vars = {
        "DASHSCOPE_API_KEY": "test-key",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield


def test_text_embedding_client_success(mock_env):
    client = TextEmbeddingClient(api_base="https://api.example.com")

    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = {
        "data": [
            {"index": 0, "embedding": [0.1, 0.2]},
            {"index": 1, "embedding": [0.3, 0.4]},
        ]
    }

    with patch("requests.post", return_value=success_response):
        result = client.embed_texts(["a", "b"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_text_embedding_client_retries_on_429(mock_env):
    client = TextEmbeddingClient(api_base="https://api.example.com")
    TextEmbeddingClient._retry_not_before = 0.0

    rate_limit_response = MagicMock()
    rate_limit_response.status_code = 429
    rate_limit_response.text = "rate limited"
    rate_limit_response.headers = {"Retry-After": "0.2"}

    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = {
        "data": [{"index": 0, "embedding": [0.5, 0.6]}]
    }

    with patch("requests.post", side_effect=[rate_limit_response, success_response]) as mock_post:
        with patch("time.sleep") as mock_sleep:
            result = client.embed_texts(["hello"])

    assert result == [[0.5, 0.6]]
    assert mock_post.call_count == 2
    assert mock_sleep.call_count >= 1


def test_text_embedding_client_no_retry_on_400(mock_env):
    client = TextEmbeddingClient(api_base="https://api.example.com")
    TextEmbeddingClient._retry_not_before = 0.0

    bad_request_response = MagicMock()
    bad_request_response.status_code = 400
    bad_request_response.text = "bad request"
    bad_request_response.headers = {}

    with patch("requests.post", return_value=bad_request_response):
        with pytest.raises(requests.HTTPError):
            client.embed_texts(["hello"])
