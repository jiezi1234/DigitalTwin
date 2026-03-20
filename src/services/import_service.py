"""
通用导入服务
支持并行嵌入、批量写入、增量追踪
兼容 CSV (微信) 和 PDF/PPT (教材)
"""

import os
import time
import logging
import concurrent.futures
from typing import List, Dict, Any, Type, Optional
from src.infrastructure.db_client import DBClient
from src.loaders.base import DataLoader
from src.infrastructure.document import Document
from src.infrastructure.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class ImportService:
    """通用导入引擎"""

    def __init__(self, db_client: DBClient):
        self.db_client = db_client
        self.embeddings = db_client.embeddings

    def _embed_chunk(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """单个块的嵌入逻辑，含简单的指数退避重试"""
        for attempt in range(max_retries):
            try:
                return self.embeddings.embed_documents(texts)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = (2 ** (attempt + 1))
                logger.warning(f"Embedding 失败 (尝试 {attempt+1}): {e}. 重试中 {wait}s...")
                time.sleep(wait)
        return []

    def import_documents(
        self,
        loader_cls: Type[DataLoader],
        pattern: str,
        collection_name: str,
        incremental: bool = False,
        tracking_file: Optional[str] = None,
        batch_size: int = 10,  # 每次调 API 的文本数 (DashScope 限制 10)
        max_workers: int = 4,
        **loader_kwargs
    ) -> Dict[str, Any]:
        """
        执行完整的导入流程：加载 -> 并行嵌入 -> 批量写入
        """
        with tracer.start_as_current_span("import.run") as span:
            span.set_attribute("import.pattern", pattern)
            span.set_attribute("import.collection", collection_name)
            
            start_time = time.time()
            
            # 1. 加载文件 (利用基类的批量加载和增量逻辑)
            logger.info(f"开始加载文件: {pattern} (增量: {incremental})")
            documents = loader_cls.load_batch(
                pattern=pattern,
                incremental=incremental,
                tracking_file=tracking_file,
                **loader_kwargs
            )
            
            if not documents:
                logger.info("没有新文档需要导入")
                return {"count": 0, "status": "skipped"}

            # 2. 分块并行生成嵌入
            num_docs = len(documents)
            logger.info(f"正在为 {num_docs} 条文档生成嵌入向量 (并发: {max_workers})...")
            
            # 将文档分成固定大小的块
            chunks = [documents[i:i + batch_size] for i in range(0, num_docs, batch_size)]
            all_embeddings = [None] * len(chunks)
            
            try:
                from tqdm import tqdm
                progress_bar = tqdm(total=len(chunks), desc="生成嵌入向量")
            except ImportError:
                progress_bar = None

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(self._embed_chunk, [d.content for d in chunk]): i 
                    for i, chunk in enumerate(chunks)
                }
                
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        all_embeddings[idx] = future.result()
                        if progress_bar:
                            progress_bar.update(1)
                    except Exception as e:
                        logger.error(f"块 {idx} 嵌入严重失败: {e}")
                        # 这里可以选择跳过或中止。目前选择抛出异常。
                        raise

            if progress_bar:
                progress_bar.close()

            # 3. 拍平结果并写入数据库
            flat_docs = []
            flat_vecs = []
            flat_metas = []
            flat_ids = []
            
            for i, chunk_vecs in enumerate(all_embeddings):
                if chunk_vecs is None: continue
                chunk_docs = chunks[i]
                for doc, vec in zip(chunk_docs, chunk_vecs):
                    flat_docs.append(doc.content)
                    flat_vecs.append(vec)
                    flat_metas.append(doc.metadata)
                    # 使用基类统一生成的 doc_id (由 base.py 中的 generate_hash 处理)
                    # 如果 Document 没 ID，则重新计算一个
                    doc_id = doc.doc_id or DataLoader.generate_hash(doc.content, doc.metadata)
                    flat_ids.append(doc_id)

            # 4. 调用 bulk_import 一次性写入
            logger.info(f"写入向量数据库 (集合: {collection_name})...")
            import_count = self.db_client.bulk_import(
                collection_name=collection_name,
                documents=flat_docs,
                embeddings=flat_vecs,
                metadatas=flat_metas,
                ids=flat_ids
            )
            
            duration = time.time() - start_time
            logger.info(f"导入完成! 耗时: {duration:.2f}s, 共导入 {import_count} 条记录")
            
            return {
                "count": import_count,
                "duration": duration,
                "status": "success",
                "documents": documents # 返回文档列表供后续分析 (如 max_tokens)
            }

    @staticmethod
    def compute_max_tokens(docs: List[Document], percentile: int = 90, scale: float = 1.5) -> int:
        """采样消息长度，估算合适的 max_tokens (主要针对聊天数据)"""
        # 仅统计 self 消息
        self_lengths = [
            len(d.metadata.get("msg_content", d.content)) # 如果没 msg_content 就用 content
            for d in docs
            if str(d.metadata.get("is_sender", "0")) == "1"
        ]
        
        if not self_lengths:
            return 150
            
        self_lengths.sort()
        idx = min(int(len(self_lengths) * percentile / 100), len(self_lengths) - 1)
        p_val = self_lengths[idx]
        result = max(50, int(p_val * scale))
        logger.info(f"消息长度分析: P{percentile}={p_val}, 建议 max_tokens={result}")
        return result
