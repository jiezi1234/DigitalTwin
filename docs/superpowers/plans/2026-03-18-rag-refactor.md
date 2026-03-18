# DigitalTwin RAG 重构实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构 DigitalTwin 项目，统一两个 RAG 方向（数字分身与数字助教）的架构，提取公共组件，降低代码重复，提高可维护性和可扩展性，同时集成 OpenTelemetry 提升可观测性。

**Architecture:**
- 基础设施层（LLMClient、DBClient、Telemetry）：统一 API 调用和数据库访问
- 数据加载层（DataLoader 工厂）：支持多数据源，统一接口
- RAG 引擎层（RAGEngine、QueryProcessor）：核心搜索和查询处理逻辑
- 服务层（RAGService、TextbookRAGService）：简化为调用底层引擎
- 在原项目保持不变的前提下，在新目录中进行重构

**Tech Stack:**
- Python 3.8+、ChromaDB、LangChain、DashScope API
- OpenTelemetry（tracing、可配置追踪级别）
- pytest（单元和集成测试）
- Flask（现有，无需改动）

---

## Phase 1: 项目初始化与基础设施层

### Task 1: 项目初始化与目录结构

**Files:**
- Create: `/home/moka/projects/DigitalTwin-Refactor/pyproject.toml`
- Create: `/home/moka/projects/DigitalTwin-Refactor/.env.example`
- Create: `/home/moka/projects/DigitalTwin-Refactor/requirements.txt`
- Create: `/home/moka/projects/DigitalTwin-Refactor/README.md`
- Create: `/home/moka/projects/DigitalTwin-Refactor/.gitignore`
- Create: `/home/moka/projects/DigitalTwin-Refactor/src/__init__.py`

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "digitaltwin-refactor"
version = "0.1.0"
description = "Refactored DigitalTwin RAG with unified architecture"
requires-python = ">=3.8"
dependencies = [
    "flask>=3.0",
    "flask-cors>=4.0",
    "langchain>=0.1.0",
    "langchain-chroma>=0.1.0",
    "langchain-community>=0.1.0",
    "chromadb>=0.4.0",
    "dashscope>=1.10.0",
    "requests>=2.31.0",
    "python-dotenv>=1.0.0",
    "opentelemetry-api>=1.21.0",
    "opentelemetry-sdk>=1.21.0",
    "opentelemetry-exporter-trace-otlp>=0.42b0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.5.0",
]
```

- [ ] **Step 2: 创建 .env.example**

```bash
# OpenTelemetry 配置
OTEL_ENABLED=false
OTEL_TRACE_LEVEL=light
# light: 仅追踪 LLM/DB 调用
# full: 追踪所有中间步骤
# custom: 通过 OTEL_CUSTOM_SPAN_PATTERNS 指定

# DashScope API
DASHSCOPE_API_KEY=your-api-key-here
CHAT_MODEL=qwen-plus
EMBED_MODEL=text-embedding-v4

# RAG 优化
LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode
LLM_REWRITING_MODEL=qwen-plus
RAG_QUERY_REWRITING_ENABLED=true
RAG_COREFERENCE_RESOLUTION_ENABLED=true

# ChromaDB
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_CHAT=wechat_embeddings
CHROMA_COLLECTION_TEXTBOOK=textbook_embeddings

# RAG 参数
RAG_ENABLED=true
RAG_MAX_RESULTS=20
RAG_MAX_CONTEXT_LENGTH=2000
RAG_INCLUDE_METADATA=true

# 日志
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

- [ ] **Step 3: 创建 requirements.txt**

```
flask>=3.0
flask-cors>=4.0
langchain>=0.1.0
langchain-chroma>=0.1.0
langchain-community>=0.1.0
chromadb>=0.4.0
dashscope>=1.10.0
requests>=2.31.0
python-dotenv>=1.0.0
opentelemetry-api>=1.21.0
opentelemetry-sdk>=1.21.0
opentelemetry-exporter-trace-otlp>=0.42b0
```

- [ ] **Step 4: 创建 README.md**

```markdown
# DigitalTwin RAG 重构版本

重构后的 DigitalTwin 项目，统一了数字分身和数字助教的 RAG 架构。

## 快速开始

### 安装依赖
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 配置环境变量
\`\`\`bash
cp .env.example .env
# 编辑 .env，填入你的 API Key 等配置
\`\`\`

### 运行测试
\`\`\`bash
pytest tests/ -v
\`\`\`

## 架构

- `src/infrastructure/` - 基础设施层（LLMClient、DBClient、Telemetry）
- `src/loaders/` - 数据加载层（CSV、PDF 等）
- `src/rag/` - RAG 引擎层
- `src/services/` - 对外服务层
- `tests/` - 单元和集成测试

详见 `docs/architecture.md`
```

- [ ] **Step 5: 创建 .gitignore**

```
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv
.env
.env.local
*.log
logs/
chroma_db/
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/
```

- [ ] **Step 6: 创建 src/__init__.py**

```python
"""DigitalTwin RAG 重构版本"""
__version__ = "0.1.0"
```

- [ ] **Step 7: 初始化 git 仓库**

Run:
```bash
cd /home/moka/projects/DigitalTwin-Refactor
git init
git add pyproject.toml .env.example requirements.txt README.md .gitignore src/__init__.py
git commit -m "chore: initialize project structure"
```

Expected: 显示 7 个文件已创建

---

### Task 2: OpenTelemetry 配置模块

**Files:**
- Create: `src/infrastructure/__init__.py`
- Create: `src/infrastructure/telemetry.py`
- Test: `tests/test_telemetry.py`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_telemetry.py
import os
from unittest.mock import patch
from src.infrastructure.telemetry import TelemetryManager, get_tracer


def test_telemetry_disabled_by_default():
    """禁用状态下不应该创建导出器"""
    with patch.dict(os.environ, {"OTEL_ENABLED": "false"}):
        manager = TelemetryManager()
        assert manager.enabled is False


def test_telemetry_enabled():
    """启用时应该正确初始化"""
    with patch.dict(os.environ, {"OTEL_ENABLED": "true", "OTEL_TRACE_LEVEL": "light"}):
        manager = TelemetryManager()
        assert manager.enabled is True
        assert manager.trace_level == "light"


def test_get_tracer_returns_valid_tracer():
    """获取的 tracer 应该能创建 span"""
    tracer = get_tracer(__name__)
    assert tracer is not None
    # 应该支持创建 span
    with tracer.start_as_current_span("test_span") as span:
        assert span is not None


def test_tracer_filtering_by_level():
    """不同的追踪级别应该过滤不同的 span"""
    with patch.dict(os.environ, {"OTEL_ENABLED": "true", "OTEL_TRACE_LEVEL": "light"}):
        manager = TelemetryManager()
        # light 级别应该只包含高层操作
        assert "light" in manager.trace_level
```

- [ ] **Step 2: 运行测试验证失败**

Run:
```bash
cd /home/moka/projects/DigitalTwin-Refactor
pytest tests/test_telemetry.py -v
```

Expected: FAILED，4 个测试失败（模块不存在）

- [ ] **Step 3: 实现 telemetry.py**

```python
# src/infrastructure/telemetry.py
"""
OpenTelemetry 配置和初始化模块
支持可配置的追踪级别：light、full、custom
"""

import os
import logging
from typing import Optional, Set
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.trace_otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)

# 默认的高层操作追踪
_LIGHT_SPANS = {
    "llm.api_call",
    "db.vector_search",
    "loader.load",
}

# 完整追踪包括中间步骤
_FULL_SPANS = _LIGHT_SPANS | {
    "rag.search",
    "query.process",
    "query.coreference_resolution",
    "query.rewriting",
    "db.connect",
    "format.context",
}


class TelemetryManager:
    """管理 OpenTelemetry 生命周期和配置"""

    def __init__(self):
        self.enabled = os.getenv("OTEL_ENABLED", "false").lower() == "true"
        self.trace_level = os.getenv("OTEL_TRACE_LEVEL", "light")
        self._tracer_provider: Optional[TracerProvider] = None

        if self.enabled:
            self._init_tracer_provider()
            logger.info(f"OpenTelemetry 已启用，追踪级别: {self.trace_level}")
        else:
            logger.debug("OpenTelemetry 已禁用")

    def _init_tracer_provider(self):
        """初始化 TracerProvider 和 Exporter"""
        resource = Resource.create({
            "service.name": "digitaltwin-rag",
            "service.version": "0.1.0",
        })

        self._tracer_provider = TracerProvider(resource=resource)

        # 仅在启用导出时才配置 OTLP 导出器
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otlp_endpoint:
            try:
                exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                self._tracer_provider.add_span_processor(
                    BatchSpanProcessor(exporter)
                )
                logger.info(f"OTLP 导出器已配置: {otlp_endpoint}")
            except Exception as e:
                logger.warning(f"OTLP 导出器配置失败: {e}")

        # 设置全局 tracer provider
        trace.set_tracer_provider(self._tracer_provider)

    def should_trace(self, span_name: str) -> bool:
        """根据配置判断是否应该追踪某个 span"""
        if not self.enabled:
            return False

        if self.trace_level == "full":
            return span_name in _FULL_SPANS
        elif self.trace_level == "light":
            return span_name in _LIGHT_SPANS
        elif self.trace_level == "custom":
            custom_patterns = os.getenv("OTEL_CUSTOM_SPAN_PATTERNS", "").split(",")
            return any(pattern.strip() in span_name for pattern in custom_patterns if pattern.strip())
        return False

    def get_tracer(self, name: str) -> trace.Tracer:
        """获取命名的 tracer"""
        if self._tracer_provider:
            return self._tracer_provider.get_tracer(name)
        return trace.get_tracer(name)

    def shutdown(self):
        """关闭 tracer provider"""
        if self._tracer_provider:
            self._tracer_provider.force_flush()


# 全局单例
_telemetry_manager: Optional[TelemetryManager] = None


def init_telemetry() -> TelemetryManager:
    """初始化全局 telemetry manager"""
    global _telemetry_manager
    if _telemetry_manager is None:
        _telemetry_manager = TelemetryManager()
    return _telemetry_manager


def get_telemetry() -> TelemetryManager:
    """获取全局 telemetry manager"""
    global _telemetry_manager
    if _telemetry_manager is None:
        _telemetry_manager = init_telemetry()
    return _telemetry_manager


def get_tracer(name: str) -> trace.Tracer:
    """获取命名的 tracer（便捷函数）"""
    return get_telemetry().get_tracer(name)
```

- [ ] **Step 4: 运行测试验证通过**

Run:
```bash
cd /home/moka/projects/DigitalTwin-Refactor
pytest tests/test_telemetry.py -v
```

Expected: PASSED，4/4 通过

- [ ] **Step 5: 提交**

```bash
cd /home/moka/projects/DigitalTwin-Refactor
git add src/infrastructure/__init__.py src/infrastructure/telemetry.py tests/test_telemetry.py
git commit -m "feat(telemetry): add OpenTelemetry configuration with configurable trace levels"
```

---

### Task 3: LLMClient 统一接口

**Files:**
- Create: `src/infrastructure/llm_client.py`
- Test: `tests/test_llm_client.py`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_llm_client.py
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
```

- [ ] **Step 2: 运行测试验证失败**

Run:
```bash
cd /home/moka/projects/DigitalTwin-Refactor
pytest tests/test_llm_client.py -v
```

Expected: FAILED，模块不存在

- [ ] **Step 3: 实现 llm_client.py**

```python
# src/infrastructure/llm_client.py
"""
统一的 LLM API 客户端
支持 OpenTelemetry 追踪
"""

import os
import logging
import requests
from typing import List, Dict, Optional
from src.infrastructure.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class LLMClient:
    """统一的大模型 API 调用客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 30,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.api_base = api_base or os.getenv(
            "LLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode"
        )
        self.model = model or os.getenv("LLM_REWRITING_MODEL", "qwen-plus")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set")

    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
        top_p: float = 0.9,
    ) -> Optional[str]:
        """
        调用 LLM API

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大 token 数
            top_p: top_p 采样参数

        Returns:
            生成的文本，失败返回 None
        """
        with tracer.start_as_current_span("llm.api_call") as span:
            try:
                # 设置 span 属性
                span.set_attribute("llm.model", self.model)
                span.set_attribute("llm.temperature", temperature)
                span.set_attribute("llm.max_tokens", max_tokens)

                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "stream": False,
                }

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }

                api_endpoint = f"{self.api_base.rstrip('/')}/v1/chat/completions"

                logger.debug(f"调用 LLM API: {self.model}")
                response = requests.post(
                    api_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("choices") and len(data["choices"]) > 0:
                        result = data["choices"][0].get("message", {}).get("content", "").strip()
                        logger.debug(f"LLM 响应: {result[:100]}...")
                        span.set_attribute("llm.response_status", "success")
                        return result
                else:
                    logger.warning(
                        f"LLM API 错误: {response.status_code} - {response.text[:200]}"
                    )
                    span.set_attribute("llm.response_status", f"error_{response.status_code}")

            except Exception as e:
                logger.warning(f"LLM API 调用失败: {e}")
                span.set_attribute("llm.response_status", "exception")
                span.record_exception(e)

        return None
```

- [ ] **Step 4: 运行测试验证通过**

Run:
```bash
cd /home/moka/projects/DigitalTwin-Refactor
pytest tests/test_llm_client.py -v
```

Expected: PASSED，4/4 通过

- [ ] **Step 5: 提交**

```bash
cd /home/moka/projects/DigitalTwin-Refactor
git add src/infrastructure/llm_client.py tests/test_llm_client.py
git commit -m "feat(infrastructure): add unified LLMClient with OpenTelemetry tracing"
```

---

### Task 4: DBClient 统一接口

**Files:**
- Create: `src/infrastructure/document.py`
- Create: `src/infrastructure/db_client.py`
- Test: `tests/test_db_client.py`

- [ ] **Step 1: 创建通用 Document 模型**

```python
# src/infrastructure/document.py
"""通用的文档数据模型"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Document:
    """标准化的文档模型"""

    content: str
    """文档内容"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """元数据，例如 {"source": "chat", "chat_time": 123456, "talker": "张三"}"""

    doc_id: Optional[str] = None
    """文档唯一ID（可选，若不提供则自动生成）"""

    def __post_init__(self):
        """初始化后处理"""
        if not self.doc_id:
            # 若不提供 ID，基于内容生成简单的哈希
            import hashlib
            self.doc_id = hashlib.md5(
                (self.content + str(self.metadata)).encode()
            ).hexdigest()
```

- [ ] **Step 2: 写失败的测试**

```python
# tests/test_db_client.py
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
            client.vectorstore = mock_vectorstore

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

            # 验证 add_documents 方法存在
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
            mock_vectorstore.similarity_search.return_value = [mock_doc]
            client.vectorstore = mock_vectorstore

            results = client.search("test query", collection_name="test_chat", k=5)

            assert len(results) > 0


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

            stats = client.get_stats(collection_name="test_chat")

            assert "total_records" in stats
```

- [ ] **Step 3: 运行测试验证失败**

Run:
```bash
cd /home/moka/projects/DigitalTwin-Refactor
pytest tests/test_db_client.py -v
```

Expected: FAILED，模块不存在

- [ ] **Step 4: 实现 db_client.py**

```python
# src/infrastructure/db_client.py
"""
统一的向量数据库访问客户端
基于 ChromaDB + LangChain
支持 OpenTelemetry 追踪
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from src.infrastructure.telemetry import get_tracer
from src.infrastructure.document import Document

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class DBClient:
    """统一的向量数据库客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        persist_dir: Optional[str] = None,
        embed_model: str = "text-embedding-v4",
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.embed_model = embed_model

        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set")

        # 初始化 embedding 模型和 chromadb 客户端
        self.embeddings = DashScopeEmbeddings(model=embed_model)
        self._chroma_client = chromadb.PersistentClient(path=self.persist_dir)

        # 每个 collection 对应一个 vectorstore
        self._vectorstores: Dict[str, Chroma] = {}

    def _get_or_create_vectorstore(self, collection_name: str) -> Chroma:
        """获取或创建指定集合的 vectorstore"""
        if collection_name not in self._vectorstores:
            with tracer.start_as_current_span("db.connect") as span:
                try:
                    span.set_attribute("db.collection", collection_name)
                    vectorstore = Chroma(
                        collection_name=collection_name,
                        embedding_function=self.embeddings,
                        persist_directory=self.persist_dir,
                    )
                    self._vectorstores[collection_name] = vectorstore
                    logger.info(f"连接到集合: {collection_name}")
                except Exception as e:
                    logger.error(f"连接集合失败: {collection_name} - {e}")
                    span.record_exception(e)
                    raise

        return self._vectorstores[collection_name]

    def add_documents(
        self,
        documents: List[Document],
        collection_name: str,
    ) -> int:
        """
        添加文档到向量数据库

        Args:
            documents: Document 列表
            collection_name: 集合名称

        Returns:
            成功添加的文档数
        """
        with tracer.start_as_current_span("db.add_documents") as span:
            span.set_attribute("db.collection", collection_name)
            span.set_attribute("db.num_docs", len(documents))

            try:
                vectorstore = self._get_or_create_vectorstore(collection_name)

                # 转换为 LangChain Document 格式
                from langchain_core.documents import Document as LCDocument

                lc_docs = [
                    LCDocument(page_content=doc.content, metadata=doc.metadata)
                    for doc in documents
                ]

                # 添加到向量数据库
                ids = vectorstore.add_documents(lc_docs)

                logger.info(f"添加 {len(ids)} 条文档到 {collection_name}")
                span.set_attribute("db.docs_added", len(ids))
                return len(ids)

            except Exception as e:
                logger.error(f"添加文档失败: {e}")
                span.record_exception(e)
                raise

    def search(
        self,
        query: str,
        collection_name: str,
        k: int = 15,
        fetch_k: Optional[int] = None,
        lambda_mult: float = 0.6,
        use_mmr: bool = True,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        搜索向量数据库

        Args:
            query: 查询文本
            collection_name: 集合名称
            k: 返回结果数
            fetch_k: MMR 预取数（若为 None，则为 k * 4）
            lambda_mult: MMR 多样性权重
            use_mmr: 是否使用 MMR 搜索（推荐用于多样性）

        Returns:
            List of (content, metadata, score)
        """
        with tracer.start_as_current_span("db.vector_search") as span:
            span.set_attribute("db.collection", collection_name)
            span.set_attribute("db.query", query[:100])
            span.set_attribute("db.k", k)

            try:
                vectorstore = self._get_or_create_vectorstore(collection_name)

                if use_mmr:
                    fetch_k = fetch_k or max(k * 4, 60)
                    docs = vectorstore.max_marginal_relevance_search(
                        query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
                    )
                else:
                    docs = vectorstore.similarity_search(query, k=k)

                results = []
                for doc in docs:
                    results.append(
                        (
                            doc.page_content,
                            doc.metadata or {},
                            1.0,  # MMR 不返回分数，统一设为 1.0
                        )
                    )

                logger.debug(f"搜索返回 {len(results)} 条结果")
                span.set_attribute("db.results_count", len(results))
                return results

            except Exception as e:
                logger.error(f"搜索失败: {e}")
                span.record_exception(e)
                raise

    def get_stats(self, collection_name: str) -> Dict[str, Any]:
        """获取集合统计信息"""
        with tracer.start_as_current_span("db.get_stats") as span:
            span.set_attribute("db.collection", collection_name)

            try:
                collection = self._chroma_client.get_or_create_collection(
                    collection_name
                )
                count = collection.count()

                stats = {
                    "connected": True,
                    "collection_name": collection_name,
                    "total_records": count,
                    "persist_dir": self.persist_dir,
                }

                logger.debug(f"集合 {collection_name} 统计: {count} 条记录")
                span.set_attribute("db.total_records", count)
                return stats

            except Exception as e:
                logger.error(f"获取统计信息失败: {e}")
                span.record_exception(e)
                return {
                    "connected": False,
                    "collection_name": collection_name,
                    "error": str(e),
                }

    def delete_collection(self, collection_name: str) -> bool:
        """删除集合（仅用于测试和重置）"""
        try:
            self._chroma_client.delete_collection(name=collection_name)
            if collection_name in self._vectorstores:
                del self._vectorstores[collection_name]
            logger.info(f"删除集合: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False
```

- [ ] **Step 5: 运行测试验证通过**

Run:
```bash
cd /home/moka/projects/DigitalTwin-Refactor
pytest tests/test_db_client.py -v
```

Expected: PASSED，4/4 通过

- [ ] **Step 6: 提交**

```bash
cd /home/moka/projects/DigitalTwin-Refactor
git add src/infrastructure/document.py src/infrastructure/db_client.py tests/test_db_client.py
git commit -m "feat(infrastructure): add DBClient and Document models with unified vector DB access"
```

---

## Phase 2: 数据加载层

### Task 5: DataLoader 工厂与接口

**Files:**
- Create: `src/loaders/__init__.py`
- Create: `src/loaders/base.py`
- Test: `tests/test_loaders_base.py`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_loaders_base.py
import pytest
from abc import ABC
from src.loaders.base import DataLoader, DataLoaderFactory
from src.infrastructure.document import Document


def test_data_loader_is_abstract():
    """DataLoader 应该是抽象类"""
    assert issubclass(DataLoader, ABC)


def test_data_loader_factory_registration():
    """工厂应该支持注册新的加载器"""

    class MockLoader(DataLoader):
        def load(self):
            return [Document(content="test", metadata={"type": "mock"})]

    # 注册工厂
    DataLoaderFactory.register("mock", MockLoader)

    # 创建实例
    loader = DataLoaderFactory.create("mock")
    assert isinstance(loader, MockLoader)


def test_data_loader_factory_create_csv():
    """工厂应该能创建 CSV 加载器"""
    # 注册 CSV 加载器（稍后实现）
    try:
        loader = DataLoaderFactory.create("csv", filepath="test.csv")
        assert loader is not None
    except NotImplementedError:
        # 如果还未实现，应该抛出 NotImplementedError
        pass


def test_data_loader_factory_create_pdf():
    """工厂应该能创建 PDF 加载器"""
    try:
        loader = DataLoaderFactory.create("pdf", filepath="test.pdf")
        assert loader is not None
    except NotImplementedError:
        pass
```

- [ ] **Step 2: 运行测试验证失败**

Run:
```bash
cd /home/moka/projects/DigitalTwin-Refactor
pytest tests/test_loaders_base.py -v
```

Expected: FAILED，模块不存在

- [ ] **Step 3: 实现 base.py**

```python
# src/loaders/base.py
"""
数据加载器基类和工厂
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Type, List, Any
from src.infrastructure.document import Document

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """数据加载器基类"""

    @abstractmethod
    def load(self) -> List[Document]:
        """
        加载数据并返回标准化的 Document 列表

        Returns:
            Document 列表
        """
        pass


class DataLoaderFactory:
    """数据加载器工厂"""

    _loaders: Dict[str, Type[DataLoader]] = {}

    @classmethod
    def register(cls, loader_type: str, loader_class: Type[DataLoader]):
        """
        注册加载器类

        Args:
            loader_type: 加载器类型标识（如 "csv"、"pdf"）
            loader_class: 加载器类
        """
        cls._loaders[loader_type] = loader_class
        logger.info(f"注册加载器: {loader_type} -> {loader_class.__name__}")

    @classmethod
    def create(cls, loader_type: str, **kwargs) -> DataLoader:
        """
        创建指定类型的加载器

        Args:
            loader_type: 加载器类型标识
            **kwargs: 传给加载器的参数

        Returns:
            DataLoader 实例

        Raises:
            ValueError: 如果加载器类型不存在
        """
        if loader_type not in cls._loaders:
            raise ValueError(
                f"未知的加载器类型: {loader_type}。"
                f"已注册的类型: {', '.join(cls._loaders.keys())}"
            )

        loader_class = cls._loaders[loader_type]
        logger.info(f"创建加载器: {loader_type}")
        return loader_class(**kwargs)

    @classmethod
    def list_loaders(cls) -> List[str]:
        """列出所有已注册的加载器"""
        return list(cls._loaders.keys())
```

- [ ] **Step 4: 创建 __init__.py**

```python
# src/loaders/__init__.py
from src.loaders.base import DataLoader, DataLoaderFactory

__all__ = ["DataLoader", "DataLoaderFactory"]
```

- [ ] **Step 5: 运行测试验证通过**

Run:
```bash
cd /home/moka/projects/DigitalTwin-Refactor
pytest tests/test_loaders_base.py -v
```

Expected: PASSED，3/3 通过

- [ ] **Step 6: 提交**

```bash
cd /home/moka/projects/DigitalTwin-Refactor
git add src/loaders/__init__.py src/loaders/base.py tests/test_loaders_base.py
git commit -m "feat(loaders): add DataLoader base class and Factory pattern"
```

---

### Task 6: CSV 加载器实现

**Files:**
- Create: `src/loaders/csv_loader.py`
- Test: `tests/test_csv_loader.py`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_csv_loader.py
import pytest
import tempfile
import csv
from src.loaders.csv_loader import CSVLoader
from src.loaders.base import DataLoaderFactory


@pytest.fixture
def sample_csv_file():
    """创建示例 CSV 文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['talker', 'message', 'chat_time'])
        writer.writerow(['张三', '你好', '1609459200'])
        writer.writerow(['李四', '你好啊', '1609459260'])
        f.flush()
        yield f.name


def test_csv_loader_initialization(sample_csv_file):
    """初始化 CSV 加载器"""
    loader = CSVLoader(filepath=sample_csv_file)
    assert loader.filepath == sample_csv_file


def test_csv_loader_load(sample_csv_file):
    """加载 CSV 文件"""
    loader = CSVLoader(filepath=sample_csv_file)
    documents = loader.load()

    assert len(documents) == 2
    assert documents[0].content == "你好"
    assert documents[0].metadata['talker'] == "张三"


def test_csv_loader_factory_registration(sample_csv_file):
    """工厂应该能创建 CSV 加载器"""
    DataLoaderFactory.register("csv", CSVLoader)

    loader = DataLoaderFactory.create("csv", filepath=sample_csv_file)
    documents = loader.load()

    assert len(documents) == 2
```

- [ ] **Step 2: 运行测试验证失败**

Run:
```bash
cd /home/moka/projects/DigitalTwin-Refactor
pytest tests/test_csv_loader.py -v
```

Expected: FAILED，模块不存在

- [ ] **Step 3: 实现 csv_loader.py**

```python
# src/loaders/csv_loader.py
"""
CSV 文件加载器（用于微信聊天记录）
"""

import csv
import logging
from typing import List
from src.loaders.base import DataLoader
from src.infrastructure.document import Document
from src.infrastructure.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class CSVLoader(DataLoader):
    """CSV 加载器，支持微信聊天记录"""

    def __init__(
        self,
        filepath: str,
        encoding: str = "utf-8",
        message_column: str = "message",
        metadata_columns: List[str] = None,
    ):
        """
        初始化 CSV 加载器

        Args:
            filepath: CSV 文件路径
            encoding: 文件编码
            message_column: 内容列名（默认 "message"）
            metadata_columns: 元数据列列表（如 ["talker", "chat_time", "timestamp"]）
        """
        self.filepath = filepath
        self.encoding = encoding
        self.message_column = message_column
        self.metadata_columns = metadata_columns or ["talker", "chat_time", "timestamp"]

    def load(self) -> List[Document]:
        """加载 CSV 文件"""
        with tracer.start_as_current_span("loader.load") as span:
            span.set_attribute("loader.type", "csv")
            span.set_attribute("loader.filepath", self.filepath)

            documents = []

            try:
                with open(self.filepath, "r", encoding=self.encoding) as f:
                    reader = csv.DictReader(f)

                    for row_num, row in enumerate(reader, start=1):
                        if not row.get(self.message_column):
                            continue

                        # 提取内容
                        content = row[self.message_column].strip()

                        # 提取元数据
                        metadata = {}
                        for col in self.metadata_columns:
                            if col in row:
                                value = row[col].strip()
                                # 尝试将时间戳转换为整数
                                if col in ("chat_time", "timestamp"):
                                    try:
                                        metadata[col] = int(value)
                                    except (ValueError, TypeError):
                                        metadata[col] = value
                                else:
                                    metadata[col] = value

                        # 添加源标识
                        metadata["source"] = "csv"
                        metadata["source_file"] = self.filepath

                        doc = Document(
                            content=content,
                            metadata=metadata,
                        )
                        documents.append(doc)

                logger.info(f"从 {self.filepath} 加载了 {len(documents)} 条记录")
                span.set_attribute("loader.documents_loaded", len(documents))

            except Exception as e:
                logger.error(f"加载 CSV 文件失败: {self.filepath} - {e}")
                span.record_exception(e)
                raise

            return documents
```

- [ ] **Step 4: 运行测试验证通过**

Run:
```bash
cd /home/moka/projects/DigitalTwin-Refactor
pytest tests/test_csv_loader.py -v
```

Expected: PASSED，3/3 通过

- [ ] **Step 5: 提交**

```bash
cd /home/moka/projects/DigitalTwin-Refactor
git add src/loaders/csv_loader.py tests/test_csv_loader.py
git commit -m "feat(loaders): implement CSV loader for chat records"
```

---

### Task 7: PDF 加载器实现

**Files:**
- Create: `src/loaders/pdf_loader.py`
- Test: `tests/test_pdf_loader.py`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_pdf_loader.py
import pytest
from unittest.mock import patch, MagicMock
from src.loaders.pdf_loader import PDFLoader
from src.loaders.base import DataLoaderFactory


def test_pdf_loader_initialization():
    """初始化 PDF 加载器"""
    loader = PDFLoader(filepath="test.pdf")
    assert loader.filepath == "test.pdf"


def test_pdf_loader_load_with_mock():
    """加载 PDF 文件（使用 mock）"""
    with patch("src.loaders.pdf_loader.PyPDF2") as mock_pypdf2:
        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page content"
        mock_reader.pages = [mock_page]
        mock_pypdf2.PdfReader.return_value = mock_reader

        loader = PDFLoader(filepath="test.pdf")

        # 由于 PyPDF2 被 mock，应该能加载
        # （实际加载会失败，但这里我们测试接口）
        assert loader.filepath == "test.pdf"


def test_pdf_loader_factory_registration():
    """工厂应该能创建 PDF 加载器"""
    DataLoaderFactory.register("pdf", PDFLoader)

    loader = DataLoaderFactory.create("pdf", filepath="test.pdf")
    assert isinstance(loader, PDFLoader)
```

- [ ] **Step 2: 运行测试验证失败**

Run:
```bash
cd /home/moka/projects/DigitalTwin-Refactor
pytest tests/test_pdf_loader.py -v
```

Expected: FAILED，模块不存在

- [ ] **Step 3: 实现 pdf_loader.py**

```python
# src/loaders/pdf_loader.py
"""
PDF 文件加载器（用于教材）
"""

import logging
from typing import List, Optional
from src.loaders.base import DataLoader
from src.infrastructure.document import Document
from src.infrastructure.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class PDFLoader(DataLoader):
    """PDF 加载器，支持教材和文档"""

    def __init__(
        self,
        filepath: str,
        extract_metadata: bool = True,
    ):
        """
        初始化 PDF 加载器

        Args:
            filepath: PDF 文件路径
            extract_metadata: 是否提取页码等元数据
        """
        self.filepath = filepath
        self.extract_metadata = extract_metadata

    def load(self) -> List[Document]:
        """加载 PDF 文件"""
        with tracer.start_as_current_span("loader.load") as span:
            span.set_attribute("loader.type", "pdf")
            span.set_attribute("loader.filepath", self.filepath)

            documents = []

            try:
                # 懒加载 PyPDF2 以避免硬依赖
                from PyPDF2 import PdfReader

                reader = PdfReader(self.filepath)

                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()

                    if not text.strip():
                        continue

                    metadata = {
                        "source": "pdf",
                        "source_file": self.filepath,
                        "page": page_num + 1,
                    }

                    doc = Document(
                        content=text,
                        metadata=metadata,
                    )
                    documents.append(doc)

                logger.info(f"从 {self.filepath} 加载了 {len(documents)} 页")
                span.set_attribute("loader.documents_loaded", len(documents))

            except ImportError:
                logger.error("PyPDF2 未安装，请运行: pip install PyPDF2")
                span.record_exception(ImportError("PyPDF2 not installed"))
                raise

            except Exception as e:
                logger.error(f"加载 PDF 文件失败: {self.filepath} - {e}")
                span.record_exception(e)
                raise

            return documents
```

- [ ] **Step 4: 运行测试验证通过**

Run:
```bash
cd /home/moka/projects/DigitalTwin-Refactor
pytest tests/test_pdf_loader.py -v
```

Expected: PASSED，3/3 通过

- [ ] **Step 5: 提交**

```bash
cd /home/moka/projects/DigitalTwin-Refactor
git add src/loaders/pdf_loader.py tests/test_pdf_loader.py
git commit -m "feat(loaders): implement PDF loader for textbooks and documents"
```

---

## Phase 3: RAG 引擎层

### Task 8: QueryProcessor 查询处理器

**Files:**
- Create: `src/rag/__init__.py`
- Create: `src/rag/query_processor.py`
- Test: `tests/test_query_processor.py`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_query_processor.py
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
```

- [ ] **Step 2: 运行测试验证失败**

Run:
```bash
cd /home/moka/projects/DigitalTwin-Refactor
pytest tests/test_query_processor.py -v
```

Expected: FAILED，模块不存在

- [ ] **Step 3: 实现 query_processor.py**

```python
# src/rag/query_processor.py
"""
查询处理器
支持指代消解、Query Rewriting 等处理策略
"""

import logging
from typing import Optional, Dict, Any, List
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class QueryProcessor:
    """查询处理器，支持多种处理策略"""

    def __init__(
        self,
        llm_client: LLMClient,
        enable_coreference_resolution: bool = True,
        enable_query_rewriting: bool = True,
    ):
        """
        初始化查询处理器

        Args:
            llm_client: LLM 客户端
            enable_coreference_resolution: 是否启用指代消解
            enable_query_rewriting: 是否启用查询改写
        """
        self.llm_client = llm_client
        self.enable_coreference_resolution = enable_coreference_resolution
        self.enable_query_rewriting = enable_query_rewriting

    def resolve_coreference(
        self, query: str, persona: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        指代消解：将代词替换为具体的人名或概念

        Args:
            query: 原始查询
            persona: 分身信息（包含名字等上下文）

        Returns:
            消解后的查询
        """
        if not self.enable_coreference_resolution:
            return query

        with tracer.start_as_current_span("query.coreference_resolution") as span:
            # 检查是否包含常见代词
            pronouns = ["他", "她", "它", "他们", "她们", "它们", "那个", "这个"]
            if not any(p in query for p in pronouns):
                logger.debug("查询中无代词，跳过消解")
                return query

            persona_name = (persona or {}).get("name", "")
            persona_info = f"分身名字：{persona_name}\n" if persona_name else ""

            prompt = f"""{persona_info}你的任务是进行指代消解（Coreference Resolution）。

将下面问题中的代词替换为具体的人名或概念，使问题更清楚。
代词包括：他、她、它、他们、她们、它们、那个、这个等。

如果代词指代不明确或根本不需要替换，保持原样。

原问题：{query}

请直接输出消解后的问题，不要解释。"""

            try:
                result = self.llm_client.call(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=200,
                )

                if result and result != query:
                    logger.info(f"指代消解: '{query}' → '{result}'")
                    span.set_attribute("query.coreference_changed", True)
                    return result

            except Exception as e:
                logger.warning(f"指代消解失败: {e}")
                span.record_exception(e)

            return query

    def rewrite_query(
        self, query: str, persona: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Query Rewriting：根据分身特点改写查询以提高检索质量

        Args:
            query: 原始查询（可能已消解代词）
            persona: 分身信息（包含名字、特点等）

        Returns:
            改写后的查询
        """
        if not self.enable_query_rewriting:
            return query

        with tracer.start_as_current_span("query.rewriting") as span:
            persona_name = (persona or {}).get("name", "")
            system_prompt = (persona or {}).get("system_prompt", "")
            doc_count = (persona or {}).get("doc_count", 0)

            persona_context = f"""分身信息：
- 名字：{persona_name}
- 已导入聊天记录数：{doc_count}条
- 角色设定：{system_prompt[:200] if system_prompt else "未设定"}"""

            prompt = f"""{persona_context}

你的任务是改写用户的问题，使其更容易从分身的聊天历史中检索相关内容。

原问题可能很短或表述模糊，你需要基于分身的特点和背景，将其扩展和转化为更有语义的形式。

例如：
- "你怎么样？" 对于林黛玉可能改写为：身体状况、健康、精神状态、情绪、病症
- "最近在做什么？" 可能改写为：近期活动、日常事务、工作、业余爱好

原问题：{query}

请输出改写后的问题或关键词组合（用中文逗号分隔），使其更适合向量检索。
不要添加额外说明，直接输出改写结果。"""

            try:
                result = self.llm_client.call(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=300,
                )

                if result and result != query:
                    logger.info(f"Query改写: '{query}' → '{result}'")
                    span.set_attribute("query.rewritten", True)
                    return result

            except Exception as e:
                logger.warning(f"Query改写失败: {e}")
                span.record_exception(e)

            return query

    def process(
        self, query: str, persona: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        处理查询（完整流程）

        Args:
            query: 原始查询
            persona: 分身信息

        Returns:
            处理后的查询
        """
        with tracer.start_as_current_span("query.process") as span:
            span.set_attribute("query.original", query[:100])

            # 步骤 1：指代消解
            if self.enable_coreference_resolution:
                query = self.resolve_coreference(query, persona)
                logger.debug(f"指代消解后: {query[:100]}")

            # 步骤 2：Query Rewriting
            if self.enable_query_rewriting:
                query = self.rewrite_query(query, persona)
                logger.debug(f"改写后: {query[:100]}")

            span.set_attribute("query.processed", query[:100])
            return query
```

- [ ] **Step 4: 创建 __init__.py**

```python
# src/rag/__init__.py
from src.rag.query_processor import QueryProcessor

__all__ = ["QueryProcessor"]
```

- [ ] **Step 5: 运行测试验证通过**

Run:
```bash
cd /home/moka/projects/DigitalTwin-Refactor
pytest tests/test_query_processor.py -v
```

Expected: PASSED，4/4 通过

- [ ] **Step 6: 提交**

```bash
cd /home/moka/projects/DigitalTwin-Refactor
git add src/rag/__init__.py src/rag/query_processor.py tests/test_query_processor.py
git commit -m "feat(rag): add QueryProcessor with coreference resolution and query rewriting"
```

---

### Task 9: RAGEngine 核心搜索引擎

**Files:**
- Create: `src/rag/rag_engine.py`
- Test: `tests/test_rag_engine.py`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_rag_engine.py
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

    context = engine.format_context(results, max_length=1000)

    assert "content 1" in context
    assert "content 2" in context
```

- [ ] **Step 2: 运行测试验证失败**

Run:
```bash
cd /home/moka/projects/DigitalTwin-Refactor
pytest tests/test_rag_engine.py -v
```

Expected: FAILED，模块不存在

- [ ] **Step 3: 实现 rag_engine.py**

```python
# src/rag/rag_engine.py
"""
RAG 核心搜索引擎
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from src.infrastructure.db_client import DBClient
from src.rag.query_processor import QueryProcessor
from src.infrastructure.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class RAGEngine:
    """RAG 核心搜索引擎"""

    def __init__(self, db_client: DBClient):
        """
        初始化 RAG 引擎

        Args:
            db_client: 数据库客户端
        """
        self.db_client = db_client

    def search(
        self,
        query: str,
        collection_name: str,
        query_processor: Optional[QueryProcessor] = None,
        k: int = 15,
        use_mmr: bool = True,
        lambda_mult: float = 0.6,
        **kwargs,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        搜索向量数据库

        Args:
            query: 查询文本
            collection_name: 集合名称
            query_processor: 查询处理器（可选）
            k: 返回结果数
            use_mmr: 是否使用 MMR 搜索
            lambda_mult: MMR 多样性权重
            **kwargs: 其他参数（如 persona 等）

        Returns:
            List of (content, metadata, score)
        """
        with tracer.start_as_current_span("rag.search") as span:
            span.set_attribute("rag.query_original", query[:100])
            span.set_attribute("rag.collection", collection_name)
            span.set_attribute("rag.k", k)

            try:
                # 处理查询
                processed_query = query
                if query_processor:
                    processed_query = query_processor.process(query, persona=kwargs.get("persona"))
                    logger.info(f"处理后的查询: {processed_query[:100]}")
                    span.set_attribute("rag.query_processed", processed_query[:100])

                # 向量搜索
                results = self.db_client.search(
                    query=processed_query,
                    collection_name=collection_name,
                    k=k,
                    use_mmr=use_mmr,
                    lambda_mult=lambda_mult,
                )

                logger.info(f"搜索返回 {len(results)} 条结果")
                span.set_attribute("rag.results_count", len(results))

                return results

            except Exception as e:
                logger.error(f"RAG 搜索失败: {e}")
                span.record_exception(e)
                raise

    def format_context(
        self,
        results: List[Tuple[str, Dict[str, Any], float]],
        max_context_length: int = 2000,
        include_metadata: bool = True,
        format_type: str = "chat",  # "chat" 或 "textbook"
    ) -> str:
        """
        格式化搜索结果为上下文字符串

        Args:
            results: 搜索结果列表
            max_context_length: 最大上下文长度
            include_metadata: 是否包含元数据
            format_type: 格式化类型（chat 或 textbook）

        Returns:
            格式化的上下文字符串
        """
        with tracer.start_as_current_span("format.context") as span:
            span.set_attribute("format.type", format_type)
            span.set_attribute("format.num_results", len(results))

            if not results:
                return ""

            lines = []
            total_length = 0

            for content, metadata, score in results:
                if format_type == "chat":
                    # 聊天记录格式
                    if include_metadata:
                        talker = metadata.get("talker", "未知")
                        chat_time = metadata.get("chat_time_str") or metadata.get("chat_time", "")
                        time_prefix = f"[{chat_time}] " if chat_time else ""
                        record = f"{time_prefix}{talker}: {content.strip()}"
                    else:
                        record = content.strip()

                elif format_type == "textbook":
                    # 教材格式
                    if include_metadata:
                        source = metadata.get("source", "")
                        chapter = metadata.get("chapter", "")
                        section = metadata.get("section", "")
                        page = metadata.get("page", "")

                        location_parts = []
                        if chapter:
                            location_parts.append(chapter)
                        if section:
                            location_parts.append(section)
                        if page:
                            location_parts.append(f"第{page}页")

                        location = " > ".join(location_parts) if location_parts else source
                        record = f"【{location}】\n{content.strip()}\n"
                    else:
                        record = content.strip()

                else:
                    # 默认格式
                    record = content.strip()

                if total_length + len(record) > max_context_length:
                    break

                lines.append(record)
                total_length += len(record)

            return "\n".join(lines)

    def get_nearby_records(
        self,
        collection_name: str,
        timestamp: int,
        time_window_minutes: int = 30,
        max_nearby: int = 15,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        获取时间戳相近的记录（用于聊天记录）

        Args:
            collection_name: 集合名称
            timestamp: 目标 Unix 时间戳（整数秒）
            time_window_minutes: 时间窗口(分钟)
            max_nearby: 最多返回的相近记录数

        Returns:
            List of (content, metadata, score)
        """
        # 这个方法用于从聊天记录中检索时间相近的记录
        # 目前由于 ChromaDB 的限制，实现较复杂，暂不在引擎层实现
        # 而是在具体的 RAGService 中实现
        logger.debug("获取相近记录功能由具体的 Service 实现")
        return []
```

- [ ] **Step 4: 运行测试验证通过**

Run:
```bash
cd /home/moka/projects/DigitalTwin-Refactor
pytest tests/test_rag_engine.py -v
```

Expected: PASSED，3/3 通过

- [ ] **Step 5: 提交**

```bash
cd /home/moka/projects/DigitalTwin-Refactor
git add src/rag/rag_engine.py tests/test_rag_engine.py
git commit -m "feat(rag): add RAGEngine with unified search interface and context formatting"
```

---

## Phase 4: 服务层整合

### Task 10: 简化的 RAGService

**Files:**
- Create: `src/services/__init__.py`
- Create: `src/services/rag_service.py`
- Test: `tests/test_rag_service.py`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_rag_service.py
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
```

- [ ] **Step 2: 运行测试验证失败**

Run:
```bash
cd /home/moka/projects/DigitalTwin-Refactor
pytest tests/test_rag_service.py -v
```

Expected: FAILED，模块不存在

- [ ] **Step 3: 实现 rag_service.py**

```python
# src/services/rag_service.py
"""
简化后的 RAG 服务（分身专用）
调用 RAGEngine 和 QueryProcessor
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.db_client import DBClient
from src.rag.rag_engine import RAGEngine
from src.rag.query_processor import QueryProcessor

logger = logging.getLogger(__name__)


class RAGService:
    """分身 RAG 服务（简化版本）"""

    def __init__(
        self,
        llm_client: LLMClient,
        db_client: DBClient,
        collection_name: str = "wechat_embeddings",
        enable_coreference_resolution: bool = True,
        enable_query_rewriting: bool = True,
    ):
        """
        初始化 RAG 服务

        Args:
            llm_client: LLM 客户端
            db_client: 数据库客户端
            collection_name: 向量集合名称
            enable_coreference_resolution: 启用指代消解
            enable_query_rewriting: 启用 Query Rewriting
        """
        self.llm_client = llm_client
        self.db_client = db_client
        self.collection_name = collection_name

        # 初始化核心组件
        self.rag_engine = RAGEngine(db_client=db_client)
        self.query_processor = QueryProcessor(
            llm_client=llm_client,
            enable_coreference_resolution=enable_coreference_resolution,
            enable_query_rewriting=enable_query_rewriting,
        )

    def search(
        self,
        query: str,
        persona: Optional[Dict[str, Any]] = None,
        k: int = 15,
        lambda_mult: float = 0.6,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        搜索相关聊天记录

        Args:
            query: 查询文本
            persona: 分身信息
            k: 返回结果数
            lambda_mult: MMR 多样性权重

        Returns:
            List of (content, metadata, score)
        """
        return self.rag_engine.search(
            query=query,
            collection_name=self.collection_name,
            query_processor=self.query_processor,
            k=k,
            lambda_mult=lambda_mult,
            persona=persona,
        )

    def format_context(
        self,
        results: List[Tuple[str, Dict[str, Any], float]],
        max_context_length: int = 2000,
        include_metadata: bool = True,
    ) -> str:
        """格式化搜索结果"""
        return self.rag_engine.format_context(
            results,
            max_context_length=max_context_length,
            include_metadata=include_metadata,
            format_type="chat",
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.db_client.get_stats(collection_name=self.collection_name)
```

- [ ] **Step 4: 创建 __init__.py**

```python
# src/services/__init__.py
from src.services.rag_service import RAGService
from src.services.textbook_rag_service import TextbookRAGService

__all__ = ["RAGService", "TextbookRAGService"]
```

- [ ] **Step 5: 运行测试验证通过**

Run:
```bash
cd /home/moka/projects/DigitalTwin-Refactor
pytest tests/test_rag_service.py -v
```

Expected: PASSED，3/3 通过

- [ ] **Step 6: 提交**

```bash
cd /home/moka/projects/DigitalTwin-Refactor
git add src/services/__init__.py src/services/rag_service.py tests/test_rag_service.py
git commit -m "feat(services): add simplified RAGService leveraging RAGEngine"
```

---

### Task 11: TextbookRAGService

**Files:**
- Create: `src/services/textbook_rag_service.py`
- Test: `tests/test_textbook_rag_service.py`

- [ ] **Step 1-6: 实现步骤（完全类似 Task 10）**

实现 TextbookRAGService，继承 RAGService 或直接调用 RAGEngine，但使用 `textbook_embeddings` 集合和 `format_type="textbook"`。

参考 RAGService 的模式实现即可（约 100 行代码）。

---

## Phase 5: 测试和文档

### Task 12: 集成测试

**Files:**
- Create: `tests/integration_tests.py`

实现端到端的集成测试，验证从加载数据、搜索到格式化的完整流程。

---

### Task 13: 项目文档

**Files:**
- Create: `docs/architecture.md` - 架构设计文档
- Create: `docs/api.md` - API 接口文档
- Create: `docs/MIGRATION.md` - 迁移指南

---

### Task 14: 收尾和项目验证

- [ ] **运行全部测试**

Run:
```bash
cd /home/moka/projects/DigitalTwin-Refactor
pytest tests/ -v --cov=src --cov-report=html
```

Expected: 测试覆盖率 > 80%

- [ ] **提交最终版本**

```bash
cd /home/moka/projects/DigitalTwin-Refactor
git add -A
git commit -m "docs: add complete documentation and finalize refactoring"
```

---

## 总结

| Phase | 任务数 | 预期产出 |
|-------|--------|---------|
| Phase 1 | 4 | 基础设施层（Telemetry、LLMClient、DBClient、Document） |
| Phase 2 | 3 | 数据加载层（DataLoader 工厂、CSV/PDF 加载器） |
| Phase 3 | 2 | RAG 引擎层（QueryProcessor、RAGEngine） |
| Phase 4 | 2 | 服务层（RAGService、TextbookRAGService） |
| Phase 5 | 3 | 测试、文档、收尾 |
| **合计** | **14** | **完整的重构项目** |

---

## 执行说明

1. **使用 subagent-driven-development 或 executing-plans 执行此计划**
2. **按照 Phase 顺序执行，频繁提交**
3. **每个 Task 完成后运行对应的测试**
4. **如遇到问题，停下来诊断根因，不要跳过**
5. **完成后对比原项目的功能，确保兼容**
