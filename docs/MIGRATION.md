# DigitalTwin RAG 重构 - 迁移指南

## 概述

本指南帮助你从原 DigitalTwin 项目迁移到重构版本。重构版本提供了统一的接口、更好的可维护性和完整的 OpenTelemetry 追踪支持。

## 什么改变了？

### 架构改进

| 方面 | 原项目 | 重构项目 |
|------|--------|---------|
| **代码组织** | 平铺结构 | 分层结构（infrastructure, loaders, rag, services） |
| **接口统一** | 各服务独立 | 统一的 DataLoader、RAGEngine 接口 |
| **代码重复** | 高（两个 RAG 实现） | 低（共享引擎和组件） |
| **可观测性** | 基础日志 | 完整的 OpenTelemetry 追踪 |
| **文档** | 无 | 完整的架构和 API 文档 |
| **测试** | 基础测试 | 单元 + 集成测试（80%+ 覆盖率） |

### 主要变化

#### 1. 数据加载

**原项目：**
```python
# 直接调用加载函数
from src.utils.csv_loader import load_csv
documents = load_csv("data.csv")

from src.utils.doc_loader import load_pdf
documents = load_pdf("book.pdf")
```

**重构项目：**
```python
# 统一工厂接口
from src.loaders.base import DataLoaderFactory
from src.loaders.csv_loader import CSVLoader
from src.loaders.pdf_loader import PDFLoader

# 注册加载器
DataLoaderFactory.register("csv", CSVLoader)
DataLoaderFactory.register("pdf", PDFLoader)

# 通过工厂创建
csv_loader = DataLoaderFactory.create("csv", filepath="data.csv")
pdf_loader = DataLoaderFactory.create("pdf", filepath="book.pdf")

# 加载数据（返回标准化的 Document 列表）
documents = csv_loader.load()
```

**优点：**
- 统一的接口
- 易于扩展新数据源
- 运行时动态注册加载器

#### 2. RAG 搜索

**原项目：**
```python
# 两个独立的服务类
from src.core.rag_service import RAGService
from src.core.textbook_rag_service import TextbookRAGService

# 分身搜索
rag_service = RAGService()
results = rag_service.search_chat_records(query)

# 教材搜索
textbook_service = TextbookRAGService()
results = textbook_service.search_textbook(query)
```

**重构项目：**
```python
# 共享的 RAGEngine + 特定的服务
from src.services.rag_service import RAGService
from src.services.textbook_rag_service import TextbookRAGService

# 分身搜索（启用指代消解和 Query Rewriting）
rag_service = RAGService(
    llm_client=client,
    db_client=db_client
)
results = rag_service.search(query, persona=persona)

# 教材搜索（仅启用 Query Rewriting）
textbook_service = TextbookRAGService(
    llm_client=client,
    db_client=db_client
)
results = textbook_service.search(query)
```

**优点：**
- 统一的搜索接口
- 共享的核心搜索引擎
- 可配置的查询处理策略
- 自动 OpenTelemetry 追踪

#### 3. 查询处理

**原项目：**
```python
# 查询改写和指代消解分散在各处
processed = await llm_rewrite_query(query)
resolved = await llm_resolve_pronouns(query)
```

**重构项目：**
```python
# 统一的 QueryProcessor
from src.rag.query_processor import QueryProcessor

processor = QueryProcessor(
    llm_client=client,
    enable_coreference_resolution=True,
    enable_query_rewriting=True
)

# 完整的处理流程
processed_query = processor.process(query, persona=persona)

# 或单步处理
query_with_resolved_pronouns = processor.resolve_coreference(query, persona)
improved_query = processor.rewrite_query(query, persona)
```

**优点：**
- 统一的查询处理接口
- 可配置的处理策略
- 自动追踪

#### 4. 数据库访问

**原项目：**
```python
# 直接使用 ChromaDB
collection = chroma_client.get_collection(name="wechat_embeddings")
results = collection.query(query_embeddings=[...], n_results=15)
```

**重构项目：**
```python
# 统一的 DBClient 接口
from src.infrastructure.db_client import DBClient

db_client = DBClient(persist_dir="./chroma_db")

# 统一的搜索接口
results = db_client.search(
    query="查询文本",
    collection_name="wechat_embeddings",
    k=15,
    use_mmr=True
)

# 返回 List[Tuple[content, metadata, score]] 格式
```

**优点：**
- 统一的接口（易于切换数据库）
- 内置 MMR 支持
- 自动处理嵌入和搜索

### 环境变量对比

| 变量 | 原项目 | 重构项目 | 说明 |
|------|--------|---------|------|
| `DASHSCOPE_API_KEY` | 必需 | 必需 | API 密钥 |
| `CHROMA_PATH` | 自定义 | `CHROMADB_PATH` | 数据库路径 |
| - | - | `OTEL_ENABLED` | 启用 OpenTelemetry |
| - | - | `OTEL_TRACE_LEVEL` | 追踪级别 |

## 迁移步骤

### 第 1 步：更新导入

**原项目代码：**
```python
from src.core.rag_service import RAGService
from src.utils.csv_loader import load_csv
from src.utils.doc_loader import load_pdf
```

**更新为：**
```python
from src.services.rag_service import RAGService
from src.loaders.csv_loader import CSVLoader
from src.loaders.pdf_loader import PDFLoader
from src.loaders.base import DataLoaderFactory
```

### 第 2 步：更新数据加载

**原项目：**
```python
documents = load_csv("wechat_records.csv")
db_client.add_documents(documents, collection="wechat_embeddings")
```

**更新为：**
```python
from src.loaders.base import DataLoaderFactory
from src.loaders.csv_loader import CSVLoader

# 注册加载器（一次性）
DataLoaderFactory.register("csv", CSVLoader)

# 创建并使用加载器
loader = DataLoaderFactory.create("csv", filepath="wechat_records.csv")
documents = loader.load()  # 返回 Document 对象列表
db_client.add_documents(documents, collection_name="wechat_embeddings")
```

### 第 3 步：更新 RAG 搜索

**原项目：**
```python
rag_service = RAGService()
results = rag_service.search(query)
context = rag_service.format_results(results)
```

**更新为：**
```python
rag_service = RAGService(
    llm_client=llm_client,
    db_client=db_client
)
results = rag_service.search(
    query=query,
    persona={"name": "张三", "doc_count": 1000}
)
context = rag_service.format_context(results)
```

### 第 4 步：启用可观测性

**添加 .env 配置：**
```bash
OTEL_ENABLED=true
OTEL_TRACE_LEVEL=full
```

**或在代码中启用：**
```python
import os
os.environ["OTEL_ENABLED"] = "true"
os.environ["OTEL_TRACE_LEVEL"] = "full"

from src.infrastructure.telemetry import initialize_telemetry
initialize_telemetry()
```

**查看追踪数据：**
```bash
# 导出为 JSON（用于调试）
# 或使用 OpenTelemetry 导出器（如 Jaeger）
```

## 兼容性

### 数据兼容性

✅ **向后兼容**

重构项目可以直接使用原项目的 ChromaDB 数据：

```python
# 原项目的数据库路径
db_client = DBClient(persist_dir="./chroma_db")

# 直接访问原有的集合
results = db_client.search(
    query="查询",
    collection_name="wechat_embeddings"  # 原集合名称
)
```

### API 兼容性

⚠️ **部分兼容**

如果原项目有自定义的 Flask API 端点，需要更新调用方式：

**原项目 API：**
```python
POST /search
{
    "query": "你好",
    "persona_id": "123"
}
```

**更新为：**
```python
POST /search
{
    "query": "你好",
    "persona": {
        "id": "123",
        "name": "张三",
        "doc_count": 1000
    }
}
```

### 配置兼容性

⚠️ **部分兼容**

原项目的环境变量大多兼容，但有以下变化：

| 原变量 | 新变量 | 说明 |
|--------|--------|------|
| `CHROMA_PATH` | `CHROMADB_PATH` | 数据库路径重命名 |
| - | `OTEL_ENABLED` | 新增：启用追踪 |
| - | `OTEL_TRACE_LEVEL` | 新增：追踪级别 |

## 常见问题

### Q1: 如何禁用指代消解？

```python
# 分身服务：禁用指代消解
rag_service = RAGService(
    llm_client=client,
    db_client=db_client,
    enable_coreference_resolution=False  # 禁用
)
```

### Q2: 如何关闭 OpenTelemetry 追踪？

```bash
# 环境变量
export OTEL_ENABLED=false

# 或代码中
os.environ["OTEL_ENABLED"] = "false"
```

### Q3: 如何迁移原项目的自定义数据源？

1. 创建新的 Loader 类：
```python
from src.loaders.base import DataLoader

class CustomLoader(DataLoader):
    def load(self) -> List[Document]:
        # 实现你的加载逻辑
        pass
```

2. 注册到工厂：
```python
DataLoaderFactory.register("custom", CustomLoader)
```

3. 使用：
```python
loader = DataLoaderFactory.create("custom", ...)
documents = loader.load()
```

### Q4: 原项目中的 Self-RAG 怎么办？

Self-RAG 代码保持不变，位置在 `src/core/self_rag.py`。

在重构版本中，可以结合 RAGService 使用：

```python
from src.services.rag_service import RAGService
from src.core.self_rag import SelfRAG

# 先用 RAGService 搜索
rag_service = RAGService(...)
results = rag_service.search(query, persona)
context = rag_service.format_context(results)

# 再用 Self-RAG 验证和反思
self_rag = SelfRAG(llm_client)
refined_results = self_rag.reflect_and_refine(results, query)
```

### Q5: 分身管理器需要改动吗？

不需要，`PersonaManager` 完全保持不变，位置在 `src/core/persona_manager.py`。

```python
from src.core.persona_manager import PersonaManager

# 完全兼容原代码
manager = PersonaManager()
persona = manager.get_persona("persona_id")
```

## 性能对比

### 搜索速度

重构版本使用相同的底层（ChromaDB + 向量搜索），所以搜索速度基本相同，甚至由于 MMR 优化可能更快。

### 内存占用

重构版本增加了少量内存用于：
- OpenTelemetry tracer
- QueryProcessor 实例
- RAGEngine 实例

平均增加 < 50MB

### 推荐配置

```python
# 生产环境：轻量级追踪
os.environ["OTEL_TRACE_LEVEL"] = "light"

# 开发环境：完整追踪
os.environ["OTEL_TRACE_LEVEL"] = "full"

# 调试环境：自定义追踪
os.environ["OTEL_TRACE_LEVEL"] = "custom"
os.environ["OTEL_CUSTOM_SPAN_PATTERNS"] = "llm.*,db.*"
```

## 支持

遇到问题？

1. 检查 [架构设计文档](./architecture.md)
2. 查阅 [API 接口文档](./api.md)
3. 运行测试：`pytest tests/ -v`
4. 检查日志和追踪数据
