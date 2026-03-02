"""
RAG向量数据库服务模块
使用 ChromaDB 本地向量数据库，无需独立数据库服务
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import dashscope
from langchain_chroma import Chroma
from langchain_community.embeddings.dashscope import DashScopeEmbeddings


class RAGService:
    """RAG向量数据库服务类（基于 ChromaDB 本地存储）"""

    def __init__(
        self,
        dashscope_api_key: str,
        collection_name: str = "wechat_embeddings",
        persist_directory: str = "./chroma_db",
        # 以下参数保留以兼容旧调用，但不再使用
        opengauss_host: str = None,
        opengauss_port: int = None,
        opengauss_db: str = None,
        opengauss_user: str = None,
        opengauss_password: str = None,
        embedding_dimension: int = None,
    ):
        """
        初始化RAG服务

        Args:
            dashscope_api_key: DashScope API密钥
            collection_name: ChromaDB 集合名称
            persist_directory: ChromaDB 持久化目录（本地文件夹路径）
        """
        # 设置DashScope API密钥
        os.environ["DASHSCOPE_API_KEY"] = dashscope_api_key
        dashscope.api_key = dashscope_api_key

        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # 初始化embedding模型
        self.embeddings = DashScopeEmbeddings(model="text-embedding-v3")

        # 初始化向量数据库
        self.vectorstore: Optional[Chroma] = None
        self._connect()

    def _connect(self):
        """连接（加载）本地 ChromaDB 向量数据库"""
        try:
            print(f"🔗 正在加载 ChromaDB 本地向量数据库 (目录: {self.persist_directory})...")

            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )

            # 测试连接：获取集合中文档数量
            count = self.vectorstore._collection.count()
            print(f"✅ 成功连接到 ChromaDB，当前集合中共有 {count} 条记录")

        except Exception as e:
            print(f"❌ 连接 ChromaDB 失败: {e}")
            import traceback
            traceback.print_exc()
            self.vectorstore = None
            raise ConnectionError(f"无法连接到 ChromaDB: {e}")

    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self.vectorstore is not None

    def _get_nearby_records(
        self,
        timestamp: str,
        time_window_minutes: int = 30,
        max_nearby: int = 15
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        获取时间戳相近的聊天记录

        Args:
            timestamp: 目标时间戳
            time_window_minutes: 时间窗口(分钟,默认30分钟)
            max_nearby: 最多返回的相近记录数(默认15条)

        Returns:
            List of (content, metadata, similarity_score)
        """
        if not self.is_connected():
            return []

        try:
            # 解析时间戳
            try:
                target_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except Exception:
                return []

            # 计算时间窗口边界
            start_time = target_time - timedelta(minutes=time_window_minutes)
            end_time = target_time + timedelta(minutes=time_window_minutes)

            # 使用 ChromaDB 的 where 过滤器进行时间范围查询
            # ChromaDB 元数据过滤器：使用 $gte/$lte 比较字符串格式的时间
            try:
                results = self.vectorstore.get(
                    where={
                        "$and": [
                            {"chat_time": {"$gte": start_time.isoformat()}},
                            {"chat_time": {"$lte": end_time.isoformat()}},
                        ]
                    },
                    limit=max_nearby * 2,  # 多取一些，再按时间排序截取
                    include=["documents", "metadatas"],
                )
            except Exception:
                # 若 ChromaDB 不支持该过滤器格式，退化为全量扫描
                results = self.vectorstore.get(
                    include=["documents", "metadatas"],
                )

            documents = results.get("documents", []) or []
            metadatas = results.get("metadatas", []) or []

            nearby_records = []
            for content, metadata in zip(documents, metadatas):
                if metadata and 'chat_time' in metadata:
                    try:
                        chat_time = datetime.fromisoformat(
                            metadata['chat_time'].replace('Z', '+00:00')
                        )
                        if start_time <= chat_time <= end_time:
                            time_diff = abs((target_time - chat_time).total_seconds())
                            score = 1.0 - (time_diff / (time_window_minutes * 60))
                            nearby_records.append((
                                content,
                                metadata,
                                max(0.0, score)
                            ))
                    except Exception:
                        continue

            # 按时间顺序排序并限制数量
            nearby_records.sort(key=lambda x: x[1].get('chat_time', ''))
            return nearby_records[:max_nearby]

        except Exception as e:
            print(f"⚠️ 获取相近记录失败: {e}")
            return []

    def search(
        self,
        query: str,
        k: int = 15,
        similarity_threshold: float = 0.0,
        include_nearby: bool = True,
        time_window_minutes: int = 30,
        nearby_per_result: int = 8,
        max_total_results: int = 50
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        搜索相关聊天记录，并包含时间戳相近的记录

        Args:
            query: 查询文本
            k: 初始相似度搜索返回结果数量(默认15)
            similarity_threshold: 相似度阈值（0-1）
            include_nearby: 是否包含时间相近的记录
            time_window_minutes: 时间窗口(分钟,默认30分钟)
            nearby_per_result: 每个相似结果附近获取的记录数(默认8条)
            max_total_results: 最大返回记录数(默认50,上限50)

        Returns:
            List of (content, metadata, similarity_score)
        """
        if not self.is_connected():
            raise RuntimeError("向量数据库未连接")

        if not query.strip():
            return []

        try:
            # 执行相似度搜索，ChromaDB 返回的 score 是"距离"（越小越相似），需要转换
            results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)

            formatted_results = []
            seen_ids = set()  # 用于去重

            for doc, score in results:
                # relevance_scores 已经是 [0, 1]（越大越相似）
                if score >= similarity_threshold:
                    doc_id = doc.metadata.get('id', doc.page_content[:50])
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        formatted_results.append((
                            doc.page_content,
                            doc.metadata,
                            float(score)
                        ))

                    # 如果启用了 nearby 功能，获取时间相近的记录
                    if include_nearby and 'chat_time' in doc.metadata:
                        nearby_records = self._get_nearby_records(
                            timestamp=doc.metadata['chat_time'],
                            time_window_minutes=time_window_minutes,
                            max_nearby=nearby_per_result
                        )

                        for content, metadata, nearby_score in nearby_records:
                            nearby_id = metadata.get('id', content[:50])
                            if nearby_id not in seen_ids:
                                seen_ids.add(nearby_id)
                                formatted_results.append((
                                    content,
                                    metadata,
                                    nearby_score * 0.5  # 降低时间相近记录的分数权重
                                ))

            # 按相似度分数排序并限制总数量
            formatted_results.sort(key=lambda x: x[2], reverse=True)
            return formatted_results[:min(max_total_results, 50)]

        except Exception as e:
            print(f"⚠️ RAG搜索异常: {e}")
            import traceback
            traceback.print_exc()
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
                sender = metadata.get('sender', '未知')
                chat_time = metadata.get('chat_time', '未知时间')
                record = f"【{sender}】({chat_time}): {content.strip()}"
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
            count = self.vectorstore._collection.count()

            # 获取样本数据统计发送者等信息
            sample_results = self.vectorstore.get(
                limit=min(200, count),
                include=["metadatas"],
            )
            metadatas = sample_results.get("metadatas", []) or []

            senders = set()
            msg_types = set()
            for metadata in metadatas:
                if metadata:
                    if 'sender' in metadata:
                        senders.add(metadata['sender'])
                    if 'msg_type' in metadata:
                        msg_types.add(metadata['msg_type'])

            return {
                "connected": True,
                "total_records": count,
                "sample_size": len(metadatas),
                "unique_senders": list(senders),
                "message_types": list(msg_types),
                "database_host": "local",
                "database_name": f"ChromaDB ({self.persist_directory})",
            }

        except Exception as e:
            return {
                "connected": False,
                "error": str(e)
            }
