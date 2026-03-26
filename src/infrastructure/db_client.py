"""
统一的向量数据库访问客户端
基于 ChromaDB + LangChain
支持 OpenTelemetry 追踪
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.documents import Document as LCDocument
from src.infrastructure.telemetry import get_tracer, get_meter
from src.infrastructure.document import Document

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)
meter = get_meter(__name__)

# 定义指标
db_operations_total = meter.create_counter(
    "db_operations_total",
    description="Total number of database operations",
    unit="1"
)
db_operation_duration = meter.create_histogram(
    "db_operation_duration_seconds",
    description="Duration of database operations",
    unit="s"
)


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
        start_time = time.time()
        attributes = {"db.collection": collection_name, "operation": "add_documents"}
        status = "error" # Default status

        with tracer.start_as_current_span("db.add_documents") as span:
            span.set_attribute("db.collection", collection_name)
            span.set_attribute("db.num_docs", len(documents))

            try:
                vectorstore = self._get_or_create_vectorstore(collection_name)

                # 转换为 LangChain Document 格式
                lc_docs = [
                    LCDocument(page_content=doc.content, metadata=doc.metadata)
                    for doc in documents
                ]

                # 添加到向量数据库
                ids = vectorstore.add_documents(lc_docs)

                logger.info(f"添加 {len(ids)} 条文档到 {collection_name}")
                span.set_attribute("db.docs_added", len(ids))
                status = "success"
                return len(ids)

            except Exception as e:
                logger.error(f"添加文档失败: {e}")
                span.record_exception(e)
                raise
            finally:
                duration = time.time() - start_time
                attributes["status"] = status
                db_operations_total.add(1, attributes)
                db_operation_duration.record(duration, attributes)

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
        start_time = time.time()
        attributes = {"db.collection": collection_name, "operation": "search"}
        status = "error" # Default status

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
                status = "success"
                return results

            except Exception as e:
                import traceback
                logger.error(f"搜索失败: {e}\n{traceback.format_exc()}")
                span.record_exception(e)
                raise
            finally:
                duration = time.time() - start_time
                attributes["status"] = status
                db_operations_total.add(1, attributes)
                db_operation_duration.record(duration, attributes)

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

    def search_by_embedding(
        self,
        embedding: List[float],
        collection_name: str,
        k: int = 10,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """使用预计算 query embedding 直接检索"""
        start_time = time.time()
        attributes = {"db.collection": collection_name, "operation": "search_by_embedding"}
        status = "error"

        with tracer.start_as_current_span("db.vector_search_by_embedding") as span:
            span.set_attribute("db.collection", collection_name)
            span.set_attribute("db.k", k)

            try:
                collection = self._chroma_client.get_or_create_collection(name=collection_name)
                results = collection.query(
                    query_embeddings=[embedding],
                    n_results=k,
                    include=["documents", "metadatas", "distances"],
                )

                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]

                output = []
                for document, metadata, distance in zip(documents, metadatas, distances):
                    score = 1.0 / (1.0 + float(distance))
                    output.append((document, metadata or {}, score))

                status = "success"
                return output
            except Exception as e:
                logger.error(f"按向量搜索失败: {e}")
                span.record_exception(e)
                raise
            finally:
                duration = time.time() - start_time
                attributes["status"] = status
                db_operations_total.add(1, attributes)
                db_operation_duration.record(duration, attributes)

    def bulk_import(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> int:
        """
        批量导入文档、向量和元数据（绕过 LangChain 二次嵌入）
        使用 ChromaDB 客户端直接操作，实现高性能写入。

        Args:
            collection_name: 集合名称
            documents: 文本内容列表
            embeddings: 预计算的向量列表
            metadatas: 元数据字典列表
            ids: 唯一 ID 列表

        Returns:
            导入的记录总数
        """
        start_time = time.time()
        attributes = {"db.collection": collection_name, "operation": "bulk_import"}
        status = "error" # Default status

        with tracer.start_as_current_span("db.bulk_import") as span:
            span.set_attribute("db.collection", collection_name)
            span.set_attribute("db.num_docs", len(documents))

            try:
                # 获取原始 ChromaDB 集合对象
                collection = self._chroma_client.get_or_create_collection(
                    name=collection_name
                )

                # 使用 upsert 执行批量写入数据
                collection.upsert(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids,
                )

                logger.info(f"成功批量导入 {len(ids)} 条数据到 {collection_name}")
                span.set_attribute("db.docs_imported", len(ids))
                status = "success"
                return len(ids)

            except Exception as e:
                logger.error(f"块导入失败: {e}")
                span.record_exception(e)
                raise
            finally:
                duration = time.time() - start_time
                attributes["status"] = status
                db_operations_total.add(1, attributes)
                db_operation_duration.record(duration, attributes)

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
