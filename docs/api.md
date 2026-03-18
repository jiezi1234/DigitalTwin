# DigitalTwin RAG 重构 - API 接口文档

## 基础设施层 API

### LLMClient

```python
from src.infrastructure.llm_client import LLMClient

class LLMClient:
    """统一的大模型 API 调用客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        初始化 LLMClient

        Args:
            api_key: API 密钥（默认从 DASHSCOPE_API_KEY 获取）
            api_base: API 基础地址
            model: 模型名称（默认 qwen-plus）
            timeout: 请求超时时间（秒）
        """

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
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            temperature: 生成温度（0.0-1.0）
            max_tokens: 最大生成 token 数
            top_p: nucleus sampling 参数

        Returns:
            生成的文本，或 None（如果调用失败）

        Example:
            >>> client = LLMClient()
            >>> response = client.call(
            ...     messages=[{"role": "user", "content": "你好"}],
            ...     temperature=0.7,
            ...     max_tokens=200
            ... )
            >>> print(response)
            '你好！很高兴与你交互。'
        """
```

### DBClient

```python
from src.infrastructure.db_client import DBClient

class DBClient:
    """统一的向量数据库访问客户端"""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_model: str = "alibaba-embedding-v1",
    ):
        """
        初始化 DBClient

        Args:
            persist_dir: ChromaDB 持久化目录
            embedding_model: 嵌入模型
        """

    def add_documents(
        self,
        documents: List[Document],
        collection_name: str = "default",
        batch_size: int = 100,
    ) -> int:
        """
        添加文档到集合

        Args:
            documents: Document 列表
            collection_name: 目标集合名称
            batch_size: 批处理大小

        Returns:
            成功添加的文档数
        """

    def search(
        self,
        query: str,
        collection_name: str,
        k: int = 15,
        use_mmr: bool = True,
        lambda_mult: float = 0.6,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        搜索向量数据库

        Args:
            query: 查询文本
            collection_name: 集合名称
            k: 返回结果数
            use_mmr: 是否使用最大边际相关性 (MMR)
            lambda_mult: MMR 多样性权重（0-1）

        Returns:
            List of (content, metadata, similarity_score)

        Example:
            >>> client = DBClient()
            >>> results = client.search(
            ...     query="你好",
            ...     collection_name="wechat_embeddings",
            ...     k=10
            ... )
            >>> for content, metadata, score in results:
            ...     print(f"{metadata['talker']}: {content} (相似度: {score})")
        """

    def get_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        获取集合统计信息

        Args:
            collection_name: 集合名称

        Returns:
            包含 count, embedding_model 等信息的字典
        """

    def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""

    def list_collections(self) -> List[str]:
        """列出所有集合"""
```

### Document

```python
from src.infrastructure.document import Document

class Document:
    """通用文档数据模型"""

    def __init__(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            content: 文本内容
            metadata: 元数据字典
        """
        self.content = content
        self.metadata = metadata or {}
```

## 数据加载层 API

### DataLoaderFactory

```python
from src.loaders.base import DataLoaderFactory, DataLoader

class DataLoaderFactory:
    """数据加载器工厂"""

    @classmethod
    def register(cls, loader_type: str, loader_class: Type[DataLoader]) -> None:
        """
        注册加载器类

        Args:
            loader_type: 加载器类型标识（如 "csv"、"pdf"）
            loader_class: DataLoader 的子类
        """

    @classmethod
    def create(cls, loader_type: str, **kwargs) -> DataLoader:
        """
        创建指定类型的加载器

        Args:
            loader_type: 加载器类型
            **kwargs: 传给加载器的参数

        Returns:
            DataLoader 实例

        Raises:
            ValueError: 如果加载器类型不存在

        Example:
            >>> DataLoaderFactory.register("csv", CSVLoader)
            >>> loader = DataLoaderFactory.create("csv", filepath="data.csv")
        """

    @classmethod
    def list_loaders(cls) -> List[str]:
        """列出所有已注册的加载器"""
```

### CSVLoader

```python
from src.loaders.csv_loader import CSVLoader

class CSVLoader(DataLoader):
    """CSV 加载器"""

    def __init__(
        self,
        filepath: str,
        encoding: str = "utf-8",
        message_column: str = "message",
        metadata_columns: List[str] = None,
    ):
        """
        Args:
            filepath: CSV 文件路径
            encoding: 文件编码
            message_column: 内容列名
            metadata_columns: 元数据列列表
        """

    def load(self) -> List[Document]:
        """
        加载 CSV 文件

        Returns:
            Document 列表

        Example:
            >>> loader = CSVLoader(filepath="chat.csv")
            >>> docs = loader.load()
            >>> print(len(docs))
            1234
        """
```

### PDFLoader

```python
from src.loaders.pdf_loader import PDFLoader

class PDFLoader(DataLoader):
    """PDF 加载器"""

    def __init__(
        self,
        filepath: str,
        extract_metadata: bool = True,
    ):
        """
        Args:
            filepath: PDF 文件路径
            extract_metadata: 是否提取页码等元数据
        """

    def load(self) -> List[Document]:
        """
        加载 PDF 文件

        Returns:
            Document 列表（每页为一个 Document）
        """
```

## RAG 引擎层 API

### QueryProcessor

```python
from src.rag.query_processor import QueryProcessor

class QueryProcessor:
    """查询处理器"""

    def __init__(
        self,
        llm_client: LLMClient,
        enable_coreference_resolution: bool = True,
        enable_query_rewriting: bool = True,
    ):
        """
        Args:
            llm_client: LLM 客户端
            enable_coreference_resolution: 启用指代消解
            enable_query_rewriting: 启用 Query Rewriting
        """

    def resolve_coreference(
        self,
        query: str,
        persona: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        指代消解：将代词替换为具体人名

        Args:
            query: 原始查询（如"他最近怎么样"）
            persona: 分身信息（包含 name 等）

        Returns:
            消解后的查询（如"张三最近怎么样"）
        """

    def rewrite_query(
        self,
        query: str,
        persona: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Query Rewriting：改写查询以提高检索质量

        Args:
            query: 原始查询
            persona: 分身信息（包含 name, system_prompt, doc_count 等）

        Returns:
            改写后的查询
        """

    def process(
        self,
        query: str,
        persona: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        完整的查询处理流程（先消解，再改写）

        Args:
            query: 原始查询
            persona: 分身信息

        Returns:
            处理后的查询
        """
```

### RAGEngine

```python
from src.rag.rag_engine import RAGEngine

class RAGEngine:
    """RAG 核心搜索引擎"""

    def __init__(self, db_client: DBClient):
        """
        Args:
            db_client: 数据库客户端
        """

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
            use_mmr: 是否使用 MMR
            lambda_mult: MMR 多样性权重
            **kwargs: 其他参数（如 persona）

        Returns:
            List of (content, metadata, score)
        """

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
            results: 搜索结果
            max_context_length: 最大上下文长度
            include_metadata: 是否包含元数据
            format_type: 格式化类型

        Returns:
            格式化后的上下文字符串

        Example（聊天模式）:
            >>> context = engine.format_context(results, format_type="chat")
            [2021-01-01 00:00:00] 张三: 你好
            [2021-01-01 00:01:00] 李四: 你好啊

        Example（教材模式）:
            >>> context = engine.format_context(results, format_type="textbook")
            【第一章 > 基础概念 > 第1页】
            什么是基础概念...
        """
```

## 服务层 API

### RAGService

```python
from src.services.rag_service import RAGService

class RAGService:
    """分身 RAG 服务"""

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

    def format_context(
        self,
        results: List[Tuple[str, Dict[str, Any], float]],
        max_context_length: int = 2000,
        include_metadata: bool = True,
    ) -> str:
        """格式化搜索结果（自动使用聊天格式）"""

    def get_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
```

### TextbookRAGService

```python
from src.services.textbook_rag_service import TextbookRAGService

class TextbookRAGService:
    """教材 RAG 服务"""

    def __init__(
        self,
        llm_client: LLMClient,
        db_client: DBClient,
        collection_name: str = "textbook_embeddings",
        enable_query_rewriting: bool = True,
    ):
        """
        初始化教材 RAG 服务

        注意：教材服务默认禁用指代消解（教材中不需要）
        """

    def search(
        self,
        query: str,
        k: int = 15,
        lambda_mult: float = 0.6,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """搜索相关教材内容"""

    def format_context(
        self,
        results: List[Tuple[str, Dict[str, Any], float]],
        max_context_length: int = 2000,
        include_metadata: bool = True,
    ) -> str:
        """格式化搜索结果（自动使用教材格式）"""

    def get_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
```

## 环境变量

```bash
# OpenTelemetry 配置
OTEL_ENABLED=true              # 启用追踪
OTEL_TRACE_LEVEL=full          # light, full, custom

# API 配置
DASHSCOPE_API_KEY=sk-xxx       # DashScope API 密钥
LLM_API_BASE=https://...       # API 基础地址
LLM_REWRITING_MODEL=qwen-plus  # 默认模型

# 数据库配置
CHROMADB_PATH=./chroma_db      # ChromaDB 持久化目录
```

## 常见用法

### 完整示例：分身问答

```python
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.db_client import DBClient
from src.services.rag_service import RAGService
from src.loaders.csv_loader import CSVLoader

# 初始化客户端
llm_client = LLMClient()
db_client = DBClient(persist_dir="./chroma_db")

# 加载数据（一次性）
loader = CSVLoader(filepath="wechat_records.csv")
documents = loader.load()
db_client.add_documents(documents, collection_name="张三")

# 初始化 RAG 服务
rag_service = RAGService(
    llm_client=llm_client,
    db_client=db_client,
    collection_name="张三"
)

# 用户查询
query = "他最近怎么样？"
persona = {"name": "张三", "doc_count": 1000}

# 搜索相关聊天记录
results = rag_service.search(query, persona=persona)

# 格式化上下文
context = rag_service.format_context(results)

# 生成回复（结合 LLM）
response = llm_client.call(
    messages=[
        {"role": "system", "content": "你是张三的数字分身"},
        {"role": "user", "content": f"背景信息：\n{context}\n\n问题：{query}"}
    ]
)

print(response)
```

### 教材搜索示例

```python
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.db_client import DBClient
from src.services.textbook_rag_service import TextbookRAGService
from src.loaders.pdf_loader import PDFLoader

# 初始化
llm_client = LLMClient()
db_client = DBClient()

# 加载教材
loader = PDFLoader(filepath="数学教材.pdf")
documents = loader.load()
db_client.add_documents(documents, collection_name="数学")

# 初始化教材服务
textbook_service = TextbookRAGService(
    llm_client=llm_client,
    db_client=db_client,
    collection_name="数学"
)

# 学生提问
query = "什么是导数？"
results = textbook_service.search(query, k=10)
context = textbook_service.format_context(results)

# AI 助教讲解
explanation = llm_client.call(
    messages=[
        {"role": "system", "content": "你是数学助教，用简洁易懂的方式解释概念"},
        {"role": "user", "content": f"教材内容：\n{context}\n\n问题：{query}"}
    ]
)

print(explanation)
```
