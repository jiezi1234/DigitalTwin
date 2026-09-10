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

用于加载教材和文档。基于 PyMuPDF，支持原生文本提取、OCR 与图片导出。

**特性：**
- 逐页提取文本
- 页码追踪
- 懒加载依赖

### 3. RAG 引擎层 (`src/rag/`)

核心的 RAG 搜索逻辑和查询处理。

#### 3.1 QueryProcessor (`query_processor.py`)

查询处理器结合最近会话，用一次结构化模型调用生成可独立检索的问题，避免指代消解与查询改写串行调用造成额外延迟和语义漂移。

**输出：**

- `standalone_query`：结合最近会话消解指代后的独立检索查询
- `entities`：问题中的人物、地点、课程术语等实体
- `time_range`：仅在问题显式包含时间条件时生成，并依据当前日期规范化为 ISO 日期后用于元数据过滤
- 结构化响应无法解析或模型调用失败时回退到原查询

**使用示例：**
```python
processor = QueryProcessor(
    llm_client=client,
    enable_coreference_resolution=True,
    enable_query_rewriting=True,
    history_messages=6,
)

processed_query = processor.process(
    query="那是什么时候？",
    persona={"name": "张三"},
    conversation=[
        {"role": "user", "content": "我以前说过喜欢杭州吗？"},
        {"role": "assistant", "content": "你提到过杭州。"},
    ],
)
```

#### 3.2 RAGEngine (`rag_engine.py`)

核心的 RAG 搜索引擎，统一了聊天和教材搜索的逻辑。

人物检索由可组合的 `BM25Retriever` 提供关键词通道。它对中文生成单字和二元
词片段，对英文与数字按词切分，并按 collection 懒加载缓存原文索引。
RAGEngine 将 Dense/MMR 与 BM25 的候选排名通过加权 RRF 融合，避免直接比较
不同通道的原始分数；任一通道异常时自动降级到另一通道。
可组合的 `LLMReranker` 随后用一次结构化调用对 Top-N 候选做相关性评分，
再截取最终 Top-K。候选内容以不可信 JSON 数据传入；模型失败、输出缺失或
解析异常时保持 RRF 原排名，避免重排服务成为单点故障。
`MetadataFilterBuilder` 将查询理解中的 ISO 日期按配置时区转换为秒级时间戳，
并把相同的 Chroma 条件同时交给 Dense/MMR 与 BM25。无时间条件、日期无效或
起止时间倒置时不生成过滤器，防止错误约束导致整条检索链路失效。
`ContextBuilder` 在生成前对候选执行规范化去重和三元字符片段相似度检测，
同时限制单条片段与整体字符预算。超长内容优先在句子边界截断，并返回
`selected_results`，确保教材回答中的引用编号仍对应实际进入提示词的来源。

**核心方法：**

1. **search()**
   ```python
   results = engine.search(
       query="查询",
       collection_name="wechat_embeddings",
       query_processor=processor,
       k=15,
       use_mmr=True,
       lambda_mult=0.6,
       hybrid_search=True,
       hybrid_candidates=30,
       rrf_k=60,
       rerank=True,
       rerank_candidates=20,
       metadata_filtering=True,
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

3. **expand_chat_neighbors()**
   - 围绕前几个语义命中点读取同一 `conversation_id` 的前后消息
   - 使用 `message_index` 恢复对话顺序，并对重叠邻域去重
   - 旧索引缺少邻域元数据时保留原语义结果

4. **build_context()**
   - 去除重复和近重复片段
   - 控制单片段及整体上下文字符预算
   - 返回选中数、去重数、截断数和实际使用字符数

#### 3.3 ReActRetrievalRouter (`react_router.py`)

将 `retrieval_search` 作为代理可选工具。路由模型结合当前问题和最近会话，
只输出 `retrieve` 或 `respond` 动作；无法解析时默认检索。系统不保存或返回模型的推理过程。

### 4. 服务层 (`src/services/`)

对外提供的服务，调用底层引擎和组件。

#### 4.1 RAGService

分身 RAG 服务，负责聊天记录的搜索和检索。

**特性：**
- 通过 ReAct 路由按需调用检索工具
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
- 使用教材领域专用 Query Rewriting
- 使用多模态 Embedding 检索文本块与图片
- 使用文本 Embedding 检索 OCR collection
- 使用 RRF 融合两个不可直接比较分数的文本排序
- 文本与图片均无证据时由 API 直接拒答，不调用生成模型
- 通过 `CitationValidator` 校验回答中的文本引用是否指向实际上下文
- 教材格式输出

```python
service = TextbookRAGService(
    llm_client=client,
    db_client=db_client,
    text_collection_name="textbook_mm_text_embeddings",
    image_collection_name="textbook_mm_image_embeddings",
    ocr_collection_name="textbook_ocr_text_embeddings",
)

results = service.search(query="什么是基础概念")
context = service.format_context(results)
```

## 数据流

### 聊天 RAG 流程

```
用户查询
    ↓
ReActRetrievalRouter (retrieve / respond)
    ↓ retrieve                 ↓ respond
历史感知 QueryProcessor       跳过向量检索
    ↓
结构化时间范围过滤
    ↓
Dense/MMR + BM25
    ↓
加权 RRF 排名融合
    ↓
LLM Top-N 候选重排
    ↓
时间邻域扩展（同一会话的前后消息）
    ↓
上下文去重、片段限长与总预算
    ↓
LLMClient (生成回复)
    ↓
返回结果
```

### 教材 RAG 流程

```
用户查询
    ↓
历史感知 QueryProcessor
    ↓
多模态查询向量              文本查询向量
    ↓                           ↓
文本块 + 图片检索             OCR 文本检索
    ↓                           ↓
        RRF 文本排序融合
                 ↓
格式化上下文 + 图片引用
    ↓
证据为空？ ── 是 → 返回可配置的证据不足回复
    ↓ 否
LLMClient（生成讲解）
    ↓
文本引用编号校验
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
| `context.build` | 上下文去重、预算与截断 |
| `rerank.llm` | Top-N 候选重排调用 |

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
- **组件集成测试**: 使用 Mock 隔离外部服务，验证加载、查询处理、检索与上下文格式化的协作流程 (`integration_tests.py`)
- **覆盖率**: 当前作为持续改进指标记录，待补齐 API、导入脚本和 PDF 流程测试后再设置门禁
- **检索评测**: 使用脱敏 JSONL 标注集对比 baseline similarity 与 Query Rewriting + MMR，记录 Hit Rate@K、MRR@K、Recall@K 和延迟分位数

运行测试：
```bash
pytest tests/ -v --cov=src --cov-report=html
```
