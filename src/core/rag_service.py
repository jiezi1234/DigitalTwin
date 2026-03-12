"""
RAG向量数据库服务模块
使用 ChromaDB 本地向量数据库，无需独立数据库服务
"""

import os
import requests
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import chromadb
import dashscope
from langchain_chroma import Chroma
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
import logging

logger = logging.getLogger(__name__)


class RAGService:
    """RAG向量数据库服务类（基于 ChromaDB 本地存储）"""

    def __init__(
        self,
        dashscope_api_key: str,
        collection_name: str = "wechat_embeddings",
        persist_directory: str = "./chroma_db",
        embed_model: str = "text-embedding-v4",
        llm_api_base: str = None,
        llm_rewriting_model: str = None,
        query_rewriting_enabled: bool = True,
        coreference_resolution_enabled: bool = True,
    ):
        """
        初始化RAG服务

        Args:
            dashscope_api_key: DashScope API密钥
            collection_name: ChromaDB 集合名称
            persist_directory: ChromaDB 持久化目录（本地文件夹路径）
            embed_model: 嵌入模型名称
            llm_api_base: 大模型API基础URL（用于Query Rewriting和指代消解）
            llm_rewriting_model: 用于改写的大模型名称
            query_rewriting_enabled: 是否启用Query Rewriting
            coreference_resolution_enabled: 是否启用指代消解
        """
        # 设置DashScope API密钥
        os.environ["DASHSCOPE_API_KEY"] = dashscope_api_key
        dashscope.api_key = dashscope_api_key

        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # 保存LLM配置（从参数或环境变量）
        self.api_key = dashscope_api_key
        self.llm_api_base = llm_api_base or os.getenv("LLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode")
        self.llm_rewriting_model = llm_rewriting_model or os.getenv("LLM_REWRITING_MODEL", "qwen-plus")
        self.query_rewriting_enabled = query_rewriting_enabled
        self.coreference_resolution_enabled = coreference_resolution_enabled

        # 初始化embedding模型
        self.embeddings = DashScopeEmbeddings(model=embed_model)

        # 使用 chromadb 官方客户端，避免依赖 LangChain 私有属性
        self._chroma_client = chromadb.PersistentClient(path=persist_directory)

        # 初始化向量数据库
        self.vectorstore: Optional[Chroma] = None
        self._connect()

    def _connect(self):
        """连接（加载）本地 ChromaDB 向量数据库"""
        try:
            logger.info("正在加载 ChromaDB 本地向量数据库 (目录: %s)...", self.persist_directory)

            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )

            # 测试连接：通过官方客户端获取集合文档数量
            count = self._chroma_client.get_or_create_collection(self.collection_name).count()
            logger.info("成功连接到 ChromaDB，当前集合中共有 %s 条记录", count)

        except Exception as e:
            logger.error("连接 ChromaDB 失败: %s", e, exc_info=True)
            self.vectorstore = None
            raise ConnectionError(f"无法连接到 ChromaDB: {e}")

    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self.vectorstore is not None

    def _call_llm_api(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """调用LLM API进行文本生成

        Args:
            messages: 消息列表，包含role和content

        Returns:
            生成的文本，失败返回None
        """
        try:
            payload = {
                "model": self.llm_rewriting_model,
                "messages": messages,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 500,
                "stream": False,
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            # 构建完整的API endpoint
            api_base = self.llm_api_base.rstrip('/')
            endpoint = f"{api_base}/v1/chat/completions"

            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("choices") and len(data["choices"]) > 0:
                    return data["choices"][0].get("message", {}).get("content", "").strip()
            else:
                logger.warning("LLM API返回错误: %d - %s", response.status_code, response.text[:200])
        except Exception as e:
            logger.warning("LLM API调用失败: %s", e)

        return None

    def _resolve_coreference(self, query: str, persona: Optional[Dict[str, Any]] = None) -> str:
        """指代消解：将代词替换为具体的人名或概念

        例如："他怎么看？" -> "张三怎么看？"

        Args:
            query: 原始查询
            persona: 分身信息（包含名字等上下文）

        Returns:
            消解后的查询
        """
        if not self.coreference_resolution_enabled:
            return query

        # 如果问题中没有常见代词，直接返回
        pronouns = ['他', '她', '它', '他们', '她们', '它们', '那个', '那些', '这个', '这些']
        if not any(p in query for p in pronouns):
            return query

        persona_name = (persona or {}).get("name", "")
        persona_info = f"分身名字：{persona_name}\n" if persona_name else ""

        prompt = f"""{persona_info}你的任务是进行指代消解（Coreference Resolution）。

将下面问题中的代词替换为具体的人名或概念，使问题更清楚。
代词包括：他、她、它、他们、她们、它们、那个、这个等。

如果代词指代不明确或根本不需要替换，保持原样。

原问题：{query}

请直接输出消解后的问题，不要解释。"""

        messages = [{"role": "user", "content": prompt}]
        result = self._call_llm_api(messages)

        if result:
            logger.debug("指代消解: '%s' -> '%s'", query, result)
            return result

        return query

    def _rewrite_query(self, query: str, persona: Optional[Dict[str, Any]] = None) -> str:
        """Query Rewriting：根据分身特点改写查询以提高检索质量

        例如："你怎么样？" -> "身体、情绪、精神状态相关的讨论"

        Args:
            query: 原始查询（可能已消解代词）
            persona: 分身信息（包含名字、特点等）

        Returns:
            改写后的查询
        """
        if not self.query_rewriting_enabled:
            return query

        persona_name = (persona or {}).get("name", "")
        system_prompt = (persona or {}).get("system_prompt", "")
        doc_count = (persona or {}).get("doc_count", 0)

        persona_context = f"""分身信息：
- 名字：{persona_name}
- 已导入聊天记录数：{doc_count}条
- 角色设定：{system_prompt[:200]}"""  # 只用前200字

        prompt = f"""{persona_context}

你的任务是改写用户的问题，使其更容易从分身的聊天历史中检索相关内容。

原问题可能很短或表述模糊，你需要基于分身的特点和背景，将其扩展和转化为更有语义的形式。

例如：
- "你怎么样？" 对于林黛玉可能改写为：身体状况、健康、精神状态、情绪、病症
- "最近在做什么？" 可能改写为：近期活动、日常事务、工作、业余爱好

原问题：{query}

请输出改写后的问题或关键词组合（用中文逗号分隔），使其更适合向量检索。
不要添加额外说明，直接输出改写结果。"""

        messages = [{"role": "user", "content": prompt}]
        result = self._call_llm_api(messages)

        if result:
            logger.debug("Query改写: '%s' -> '%s'", query, result)
            return result

        return query

    def _get_nearby_records(
        self,
        timestamp: int,
        time_window_minutes: int = 30,
        max_nearby: int = 15
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        获取时间戳相近的聊天记录

        Args:
            timestamp: 目标 Unix 时间戳（整数秒）
            time_window_minutes: 时间窗口(分钟,默认30分钟)
            max_nearby: 最多返回的相近记录数(默认15条)

        Returns:
            List of (content, metadata, similarity_score)
        """
        if not self.is_connected():
            return []

        try:
            window_seconds = time_window_minutes * 60
            start_ts = timestamp - window_seconds
            end_ts = timestamp + window_seconds

            # 使用整数时间戳做数值范围过滤，避免字典序误判
            try:
                results = self.vectorstore.get(
                    where={
                        "$and": [
                            {"chat_time": {"$gte": start_ts}},
                            {"chat_time": {"$lte": end_ts}},
                        ]
                    },
                    limit=max_nearby * 2,
                    include=["documents", "metadatas"],
                )
            except Exception:
                # 若 ChromaDB 不支持该过滤器格式，退化为有限量采样，不做全量扫描
                logger.warning("时间范围过滤不支持，跳过 nearby 扩展")
                return []

            documents = results.get("documents", []) or []
            metadatas = results.get("metadatas", []) or []

            nearby_records = []
            for content, metadata in zip(documents, metadatas):
                if metadata and 'chat_time' in metadata:
                    try:
                        chat_ts = int(metadata['chat_time'])
                        if start_ts <= chat_ts <= end_ts:
                            time_diff = abs(timestamp - chat_ts)
                            score = 1.0 - (time_diff / window_seconds)
                            nearby_records.append((
                                content,
                                metadata,
                                max(0.0, score)
                            ))
                    except Exception:
                        continue

            # 按时间顺序排序并限制数量
            nearby_records.sort(key=lambda x: x[1].get('chat_time', 0))
            return nearby_records[:max_nearby]

        except Exception as e:
            logger.warning("获取相近记录失败: %s", e)
            return []

    def search(
        self,
        query: str,
        persona: Optional[Dict[str, Any]] = None,
        k: int = 15,
        similarity_threshold: float = 0.0,
        include_nearby: bool = True,
        time_window_minutes: int = 30,
        nearby_per_result: int = 8,
        max_total_results: int = 50,
        lambda_mult: float = 0.6,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        搜索相关聊天记录，并包含时间戳相近的记录

        Args:
            query: 查询文本
            persona: 分身信息（用于Query Rewriting和指代消解）
            k: MMR 最终返回结果数量(默认15)
            similarity_threshold: 保留参数，MMR 模式下不生效
            include_nearby: 是否包含时间相近的记录
            time_window_minutes: 时间窗口(分钟,默认30分钟)
            nearby_per_result: 每个结果附近获取的记录数(默认8条)
            max_total_results: 最大返回记录数(默认50,上限50)
            lambda_mult: MMR 多样性权重，0=最多样 1=最相关(默认0.6)

        Returns:
            List of (content, metadata, similarity_score)
        """
        if not self.is_connected():
            raise RuntimeError("向量数据库未连接")

        if not query.strip():
            return []

        try:
            # ── 查询改写流程 ──────────────────────────────────
            original_query = query
            persona_name = (persona or {}).get("name", "未知")

            logger.debug("【RAG检索】原始问题: '%s' (分身: %s)", query, persona_name)

            # 步骤1：指代消解
            if self.coreference_resolution_enabled:
                resolved_query = self._resolve_coreference(query, persona)
                if resolved_query != query:
                    logger.debug("  ✓ 指代消解: '%s' → '%s'", query, resolved_query)
                    query = resolved_query
                else:
                    logger.debug("  - 指代消解: 无代词需要消解")
            else:
                logger.debug("  - 指代消解: 已禁用")

            # 步骤2：Query Rewriting
            if self.query_rewriting_enabled:
                rewritten_query = self._rewrite_query(query, persona)
                if rewritten_query != original_query:
                    logger.debug("  ✓ Query改写: '%s' → '%s'", query, rewritten_query)
                    query = rewritten_query
                else:
                    logger.debug("  - Query改写: 无改写内容")
            else:
                logger.debug("  - Query改写: 已禁用")

            logger.debug("  → 最终查询: '%s'", query)

            # ── 向量检索 ────────────────────────────────────
            # 用 MMR 搜索替代纯相似度搜索，在相关性和多样性之间取平衡
            # fetch_k 先召回更多候选，再从中挑选差异最大的 k 条
            fetch_k = max(k * 4, 60)
            mmr_docs = self.vectorstore.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
            )

            formatted_results = []
            seen_ids = set()

            for doc in mmr_docs:
                doc_id = doc.metadata.get('id', doc.page_content[:50])
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    formatted_results.append((
                        doc.page_content,
                        {**doc.metadata, '_result_source': 'semantic'},
                        1.0
                    ))

                # 如果启用了 nearby 功能，获取时间相近的记录
                if include_nearby and 'chat_time' in doc.metadata:
                    chat_ts = doc.metadata['chat_time']
                    if isinstance(chat_ts, (int, float)) and chat_ts > 0:
                        nearby_records = self._get_nearby_records(
                            timestamp=int(chat_ts),
                            time_window_minutes=time_window_minutes,
                            max_nearby=nearby_per_result
                        )
                        for content, metadata, nearby_score in nearby_records:
                            nearby_id = metadata.get('id', content[:50])
                            if nearby_id not in seen_ids:
                                seen_ids.add(nearby_id)
                                formatted_results.append((
                                    content,
                                    {**metadata, '_result_source': 'temporal'},
                                    nearby_score * 0.5
                                ))

            # 按相似度分数排序并限制总数量
            formatted_results.sort(key=lambda x: x[2], reverse=True)
            result = formatted_results[:min(max_total_results, 50)]

            # 统计结果来源
            semantic_count = sum(1 for r in result if r[1].get('_result_source') == 'semantic')
            temporal_count = sum(1 for r in result if r[1].get('_result_source') == 'temporal')

            logger.debug("  ✓ 检索完成: 返回 %d 条结果 (语义: %d, 时间相近: %d)",
                       len(result), semantic_count, temporal_count)
            return result

        except Exception as e:
            logger.warning("RAG搜索异常: %s", e, exc_info=True)
            return []

    def format_context(
        self,
        results: List[Tuple[str, Dict[str, Any], float]],
        max_context_length: int = 2000,
        include_metadata: bool = True
    ) -> str:
        """
        格式化检索结果为上下文字符串

        Args:
            results: 搜索结果列表
            max_context_length: 最大上下文长度
            include_metadata: 是否包含元数据（发送者、时间等）

        Returns:
            格式化的上下文字符串
        """
        if not results:
            return ""

        lines = []
        total_length = 0

        for content, metadata, score in results:
            if include_metadata:
                chat_time = metadata.get('chat_time_str') or metadata.get('chat_time', '')
                time_prefix = f"[{chat_time}] " if chat_time else ""
                record = f"{time_prefix}{content.strip()}"
            else:
                record = content.strip()

            if total_length + len(record) > max_context_length:
                break

            lines.append(record)
            total_length += len(record)

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        if not self.is_connected():
            return {"error": "向量数据库未连接"}

        try:
            count = self._chroma_client.get_or_create_collection(self.collection_name).count()

            # 采样统计发送者/消息类型（数据量大时为近似值）
            sample_size = min(200, count)
            sample_results = self.vectorstore.get(
                limit=sample_size,
                include=["metadatas"],
            )
            metadatas = sample_results.get("metadatas", []) or []

            senders = set()
            for metadata in metadatas:
                if metadata:
                    if 'talker' in metadata:
                        senders.add(metadata['talker'])

            return {
                "connected": True,
                "total_records": count,
                "sample_size": len(metadatas),
                "unique_senders": list(senders),
                "is_approximate": count > sample_size,  # 采样未覆盖全量时标注近似
                "database_host": "local",
                "database_name": f"ChromaDB ({self.persist_directory})",
            }

        except Exception as e:
            return {
                "connected": False,
                "error": str(e)
            }
