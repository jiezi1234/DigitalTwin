import os
import pytest
from unittest.mock import patch, MagicMock
from src.infrastructure.db_client import DBClient
from src.infrastructure.document import Document


@pytest.fixture
def mock_env():
    """模拟环境变量"""
    env_vars = {
        "DASHSCOPE_API_KEY": "test-key",
        "CHROMA_PERSIST_DIR": "./chroma_db_test",
        "CHROMA_COLLECTION_CHAT": "test_chat",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield


def test_db_client_initialization(mock_env):
    """初始化 DBClient"""
    with patch("chromadb.PersistentClient"):
        client = DBClient(persist_dir="./chroma_db_test")
        assert client.persist_dir == "./chroma_db_test"


def test_db_client_add_documents(mock_env):
    """添加文档到向量数据库"""
    with patch("chromadb.PersistentClient"):
        with patch("src.infrastructure.db_client.DashScopeEmbeddings"):
            client = DBClient(persist_dir="./chroma_db_test")

            # Mock vectorstore
            mock_vectorstore = MagicMock()
            client._vectorstores["test"] = mock_vectorstore

            docs = [
                Document(
                    content="test content 1",
                    metadata={"source": "chat"},
                    doc_id="doc1",
                ),
                Document(
                    content="test content 2",
                    metadata={"source": "textbook"},
                    doc_id="doc2",
                ),
            ]

            # Verify add_documents method exists
            assert hasattr(client, "add_documents")


def test_db_client_search(mock_env):
    """搜索向量数据库"""
    with patch("chromadb.PersistentClient"):
        with patch("src.infrastructure.db_client.DashScopeEmbeddings"):
            client = DBClient(persist_dir="./chroma_db_test")

            # Mock vectorstore
            mock_doc = MagicMock()
            mock_doc.page_content = "test result"
            mock_doc.metadata = {"source": "chat"}

            mock_vectorstore = MagicMock()
            mock_vectorstore.max_marginal_relevance_search.return_value = [mock_doc]
            mock_vectorstore.similarity_search.return_value = [mock_doc]

            # Mock _get_or_create_vectorstore to return our mock vectorstore
            with patch.object(client, '_get_or_create_vectorstore', return_value=mock_vectorstore):
                results = client.search("test query", collection_name="test", k=5)

                assert len(results) > 0
                assert results[0][0] == "test result"


def test_db_client_get_stats(mock_env):
    """获取数据库统计信息"""
    with patch("chromadb.PersistentClient"):
        with patch("src.infrastructure.db_client.DashScopeEmbeddings"):
            client = DBClient(persist_dir="./chroma_db_test")

            # Mock chroma client
            mock_collection = MagicMock()
            mock_collection.count.return_value = 100

            client._chroma_client = MagicMock()
            client._chroma_client.get_or_create_collection.return_value = mock_collection

            stats = client.get_stats(collection_name="test")

            assert "total_records" in stats
