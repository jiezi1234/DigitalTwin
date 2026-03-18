"""
课本RAG检索服务
基于 ChromaDB 的教材向量检索，用于数字助教
"""

import os
import requests
import json
from typing import List, Dict, Any, Optional, Tuple
import chromadb
import dashscope
from langchain_chroma import Chroma
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
import logging

logger = logging.getLogger(__name__)


class TextbookRAGService:
    """课本 RAG 向量检索服务（数字助教专用）"""

    def __init__(
        self,
        dashscope_api_key: str,
        collection_name: str = "textbook_embeddings",
        persist_directory: str = "./chroma_db",
        embed_model: str = "text-embedding-v4",
        query_rewriting_enabled: bool = True,
        llm_api_base: str = None,
        llm_rewriting_model: str = None,
    ):
        os.environ["DASHSCOPE_API_KEY"] = dashscope_api_key
        dashscope.api_key = dashscope_api_key

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.api_key = dashscope_api_key
        self.llm_api_base = llm_api_base or os.getenv(
            "LLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode"
        )
        self.llm_rewriting_model = llm_rewriting_model or os.getenv(
            "LLM_REWRITING_MODEL", "qwen-plus"
        )
        self.query_rewriting_enabled = query_rewriting_enabled

        self.embeddings = DashScopeEmbeddings(model=embed_model)
        self._chroma_client = chromadb.PersistentClient(path=persist_directory)

        self.vectorstore: Optional[Chroma] = None
        self._connect()

    def _connect(self):
        """连接本地 ChromaDB"""
        try:
            logger.info("正在加载课本向量数据库 (集合: %s)...", self.collection_name)
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            count = self._chroma_client.get_or_create_collection(self.collection_name).count()
            logger.info("课本向量数据库已连接，共 %d 条记录", count)
        except Exception as e:
            logger.error("连接课本向量数据库失败: %s", e, exc_info=True)
            self.vectorstore = None
            raise ConnectionError(f"无法连接课本向量数据库: {e}")

    def is_connected(self) -> bool:
        return self.vectorstore is not None

    def _call_llm_api(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """调用 LLM API"""
        try:
            payload = {
                "model": self.llm_rewriting_model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 300,
                "stream": False,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            api_base = self.llm_api_base.rstrip("/")
            resp = requests.post(
                f"{api_base}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("choices"):
                    return data["choices"][0]["message"]["content"].strip()
            else:
                logger.warning("LLM API 错误: %d", resp.status_code)
        except Exception as e:
            logger.warning("LLM API 调用失败: %s", e)
        return None

    def _rewrite_query(self, query: str) -> str:
        """改写查询以提高教材检索质量"""
        if not self.query_rewriting_enabled:
            return query

        prompt = f"""你是一位数据库课程助教。请将学生的问题改写为更适合从教材中检索答案的形式。
扩展关键词，补充专业术语。

学生问题：{query}

直接输出改写后的查询，不要解释。"""

        result = self._call_llm_api([{"role": "user", "content": prompt}])
        if result:
            logger.debug("Query 改写: '%s' -> '%s'", query, result)
            return result
        return query

    def search(
        self,
        query: str,
        k: int = 8,
        max_total_results: int = 15,
        lambda_mult: float = 0.6,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        搜索课本中的相关内容

        Args:
            query: 查询文本
            k: MMR 返回结果数
            max_total_results: 最大结果数
            lambda_mult: MMR 多样性权重

        Returns:
            List of (content, metadata, score)
        """
        if not self.is_connected():
            raise RuntimeError("课本向量数据库未连接")

        if not query.strip():
            return []

        try:
            original_query = query
            logger.info("【助教检索】原始问题: '%s'", query)

            if self.query_rewriting_enabled:
                rewritten = self._rewrite_query(query)
                if rewritten != query:
                    logger.info("  ✓ Query 改写: '%s' → '%s'", query, rewritten)
                    query = rewritten

            fetch_k = max(k * 4, 40)
            mmr_docs = self.vectorstore.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
            )

            results = []
            seen = set()
            for doc in mmr_docs:
                # 用 source+page 做去重（避免 PDF header 重复导致误判）
                meta = doc.metadata
                doc_id = f"{meta.get('source', '')}_{meta.get('page', '')}_{hash(doc.page_content)}"
                if doc_id not in seen:
                    seen.add(doc_id)
                    results.append((
                        doc.page_content,
                        doc.metadata,
                        1.0,
                    ))

            results = results[:max_total_results]
            logger.info("  ✓ 检索完成: 返回 %d 条结果", len(results))

            # 展示检索结果原文（DEBUG 级别）
            for idx, (content, meta, score) in enumerate(results):
                chapter = meta.get("chapter", "")
                section = meta.get("section", "")
                page = meta.get("page", "")
                source = meta.get("source", "")
                logger.debug(
                    "  [检索结果 %d/%d] source=%s page=%s chapter=%s section=%s\n%s",
                    idx + 1, len(results), source, page, chapter, section,
                    content[:500] + ("..." if len(content) > 500 else "")
                )

            return results

        except Exception as e:
            logger.warning("课本检索异常: %s", e, exc_info=True)
            return []

    def format_context(
        self,
        results: List[Tuple[str, Dict[str, Any], float]],
        max_context_length: int = 4000,
    ) -> str:
        """格式化检索结果，附带来源信息"""
        if not results:
            return ""

        lines = []
        total_length = 0

        for content, metadata, score in results:
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

            if total_length + len(record) > max_context_length:
                break

            lines.append(record)
            total_length += len(record)

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.is_connected():
            return {"error": "课本向量数据库未连接"}
        try:
            count = self._chroma_client.get_or_create_collection(self.collection_name).count()
            return {
                "connected": True,
                "total_records": count,
                "collection": self.collection_name,
                "database": f"ChromaDB ({self.persist_directory})",
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}
