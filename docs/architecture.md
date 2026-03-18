# DigitalTwin RAG 重构 - 架构设计文档

## 概述

本文档描述 DigitalTwin 项目重构后的架构设计。重构的目标是统一数字分身和数字助教两个 RAG 方向，提取公共组件，降低代码重复，提高可维护性和可扩展性。

## 架构层级

### 1. 基础设施层 (`src/infrastructure/`)

负责与外部系统的集成和通用功能。

#### 1.1 LLMClient (`llm_client.py`)

统一的大模型 API 调用客户端，支持 OpenTelemetry 追踪。

**职责：**
- 调用 DashScope API（支持 Qwen 模型）
- 处理 API 错误和重试
- 记录 LLM API 调用的追踪信息

**使用示例：**
```python
from src.infrastructure.llm_client import LLMClient

client = LLMClient()
response = client.call(
    messages=[{"role": "user", "content": "你好"}],
    temperature=0.7,
    max_tokens=500
)
```

#### 1.2 DBClient (`db_client.py`)

统一的向量数据库访问客户端，基于 ChromaDB。

**职责：**
- 创建和管理向量集合
- 执行向量搜索（支持 MMR）
- 获取集合统计信息

**使用示例：**
```python
from src.infrastructure.db_client import DBClient

client = DBClient(persist_dir="./chroma_db")
results = client.search(
    query="查询文本",
    collection_name="wechat_embeddings",
    k=15
)
```

#### 1.3 Document (`document.py`)

通用的文档数据模型，统一了聊天记录和教材内容。

**结构：**
```python
class Document:
    content: str          # 文本内容
    metadata: Dict       # 元数据（如 talker, page, source 等）
```

#### 1.4 Telemetry (`telemetry.py`)

OpenTelemetry 配置和初始化，支持可配置的追踪级别。

**追踪级别：**
- `light`: 仅追踪 LLM API、数据库查询
- `full`: 细粒度追踪所有中间步骤
- `custom`: 自定义追踪模式

### 2. 数据加载层 (`src/loaders/`)

负责从各种数据源加载数据并转换为标准的 Document 格式。

#### 2.1 DataLoader 基类和工厂 (`base.py`)

**设计模式：** 工厂 + 策略模式

```python
class DataLoader(ABC):
    @abstractmethod
    def load(self) -> List[Document]:
        pass

class DataLoaderFactory:
    @classmethod
    def register(cls, loader_type: str, loader_class: Type[DataLoader]):
        # 注册新的加载器
        pass

    @classmethod
    def create(cls, loader_type: str, **kwargs) -> DataLoader:
        # 创建指定类型的加载器
        pass
```

**优点：**
- 易于扩展新的数据源
- 统一的加载器接口
- 运行时动态注册加载器

#### 2.2 CSV 加载器 (`csv_loader.py`)

用于加载微信聊天记录 CSV 文件。支持自定义列映射。

**特性：**
- 自动时间戳转换
- 灵活的元数据提取
- OpenTelemetry 追踪

#### 2.3 PDF 加载器 (`pdf_loader.py`)

用于加载教材和文档。基于 PyPDF2。

**特性：**
- 逐页提取文本
- 页码追踪
- 懒加载依赖

### 3. RAG 引擎层 (`src/rag/`)

核心的 RAG 搜索逻辑和查询处理。

#### 3.1 QueryProcessor (`query_processor.py`)

查询处理器，支持多种处理策略。

**策略：**

1. **指代消解 (Coreference Resolution)**
   - 将代词（他、她、它）替换为具体人名
   - 通过 LLM 进行消解

2. **Query Rewriting**
   - 根据分身特点改写查询
   - 将简短或模糊的查询扩展为更有语义的形式
   - 提高向量检索质量

**使用示例：**
```python
processor = QueryProcessor(
    llm_client=client,
    enable_coreference_resolution=True,
    enable_query_rewriting=True
)

processed_query = processor.process(
    query="他最近怎么样",
    persona={"name": "张三"}
)
```

#### 3.2 RAGEngine (`rag_engine.py`)

核心的 RAG 搜索引擎，统一了聊天和教材搜索的逻辑。

**核心方法：**

1. **search()**
   ```python
   results = engine.search(
       query="查询",
       collection_name="wechat_embeddings",
       query_processor=processor,
       k=15,
       use_mmr=True,
       lambda_mult=0.6
   )
   # 返回 List[Tuple[content, metadata, score]]
   ```

2. **format_context()**
   ```python
   # 聊天格式
   context = engine.format_context(
       results,
       format_type="chat"
   )
   # [时间] 张三: 你好
   # [时间] 李四: 你好啊

   # 教材格式
   context = engine.format_context(
       results,
       format_type="textbook"
   )
   # 【第一章 > 第一节 > 第1页】
   # 内容...
   ```

### 4. 服务层 (`src/services/`)

对外提供的服务，调用底层引擎和组件。

#### 4.1 RAGService

分身 RAG 服务，负责聊天记录的搜索和检索。

**特性：**
- 启用指代消解
- 启用 Query Rewriting
- 聊天格式输出

```python
service = RAGService(
    llm_client=client,
    db_client=db_client,
    collection_name="wechat_embeddings"
)

results = service.search(
    query="你最近怎么样",
    persona={"name": "张三", "doc_count": 100}
)
context = service.format_context(results)
```

#### 4.2 TextbookRAGService

助教 RAG 服务，负责教材内容的搜索和检索。

**特性：**
- 禁用指代消解（教材中不需要）
- 启用 Query Rewriting
- 教材格式输出

```python
service = TextbookRAGService(
    llm_client=client,
    db_client=db_client,
    collection_name="textbook_embeddings"
)

results = service.search(query="什么是基础概念")
context = service.format_context(results)
```

## 数据流

### 聊天 RAG 流程

```
用户查询
    ↓
QueryProcessor (指代消解 + Query Rewriting)
    ↓
RAGEngine (向量搜索 MMR)
    ↓
DBClient (ChromaDB 搜索)
    ↓
RAGEngine (格式化上下文)
    ↓
LLMClient (生成回复)
    ↓
返回结果
```

### 教材 RAG 流程

```
用户查询
    ↓
QueryProcessor (Query Rewriting)
    ↓
RAGEngine (向量搜索 MMR)
    ↓
DBClient (ChromaDB 搜索)
    ↓
RAGEngine (格式化上下文 - 教材格式)
    ↓
LLMClient (生成讲解)
    ↓
返回结果
```

## 可观测性 (OpenTelemetry)

### 追踪点

| Span | 说明 |
|------|------|
| `llm.api_call` | LLM API 调用 |
| `db.vector_search` | 向量数据库搜索 |
| `rag.search` | RAG 搜索流程 |
| `query.process` | 查询处理 |
| `query.coreference_resolution` | 指代消解 |
| `query.rewriting` | Query Rewriting |
| `loader.load` | 数据加载 |
| `format.context` | 上下文格式化 |

### 配置

```bash
export OTEL_ENABLED=true
export OTEL_TRACE_LEVEL=full  # light, full, custom
```

## 扩展指南

### 添加新的数据源

1. 继承 `DataLoader` 基类
2. 实现 `load()` 方法
3. 通过工厂注册

```python
from src.loaders.base import DataLoader, DataLoaderFactory

class MyCustomLoader(DataLoader):
    def load(self) -> List[Document]:
        # 实现加载逻辑
        pass

DataLoaderFactory.register("mycustom", MyCustomLoader)
loader = DataLoaderFactory.create("mycustom", **kwargs)
```

### 自定义查询处理策略

```python
from src.rag.query_processor import QueryProcessor

processor = QueryProcessor(
    llm_client=client,
    enable_coreference_resolution=True,
    enable_query_rewriting=False  # 禁用改写
)
```

### 使用不同的 LLM 模型

修改 `LLMClient` 初始化参数：

```python
client = LLMClient(
    model="qwen-max",  # 改用高级模型
    api_base="https://custom-endpoint",
    timeout=60
)
```

## 与原项目的对比

| 功能 | 原项目 | 重构项目 |
|------|--------|---------|
| 统一接口 | ✗ | ✓ |
| 代码重复 | 高 | 低 |
| 可观测性 | 部分 | 完整 (OTel) |
| 易于扩展 | 困难 | 容易 |
| 文档 | 无 | 完整 |
| 测试覆盖 | 低 | 高 |

## 测试策略

- **单元测试**: 各层独立测试 (test_*.py)
- **集成测试**: 端到端流程测试 (integration_tests.py)
- **覆盖率**: 目标 > 80%

运行测试：
```bash
pytest tests/ -v --cov=src --cov-report=html
```
