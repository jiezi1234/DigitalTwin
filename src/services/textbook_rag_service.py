"""
教材 RAG 服务（多模态版本）
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.db_client import DBClient
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.multimodal_embedding_client import MultiModalEmbeddingClient
from src.infrastructure.text_embedding_client import TextEmbeddingClient
from src.rag.query_processor import QueryProcessor
from src.rag.context_builder import ContextBuilder, ContextBuildResult

logger = logging.getLogger(__name__)

RE_CITE = re.compile(r"\[(\d+)\]")

SearchResult = Tuple[str, Dict[str, Any], float]


class TextbookRAGService:
    """教材 RAG 服务（文本块 + 图片双路召回）"""

    def __init__(
        self,
        llm_client: LLMClient,
        db_client: DBClient,
        text_collection_name: str = "textbook_mm_text_embeddings",
        image_collection_name: str = "textbook_mm_image_embeddings",
        ocr_collection_name: Optional[str] = None,
        enable_query_rewriting: bool = True,
        query_history_messages: int = 6,
        mm_client: Optional[MultiModalEmbeddingClient] = None,
        text_embedding_client: Optional[TextEmbeddingClient] = None,
        context_builder: Optional[ContextBuilder] = None,
    ):
        self.llm_client = llm_client
        self.db_client = db_client
        self.text_collection_name = text_collection_name
        self.image_collection_name = image_collection_name
        self.ocr_collection_name = ocr_collection_name
        self.mm_client = mm_client or MultiModalEmbeddingClient()
        self.text_embedding_client = text_embedding_client or (
            TextEmbeddingClient() if ocr_collection_name else None
        )
        self.context_builder = context_builder or ContextBuilder()
        self.query_processor = QueryProcessor(
            llm_client=llm_client,
            enable_coreference_resolution=False,
            enable_query_rewriting=enable_query_rewriting,
            domain="textbook",
            history_messages=query_history_messages,
        )

    def retrieve(
        self,
        query: str,
        text_k: int = 8,
        image_k: int = 4,
        ocr_k: int = 8,
        conversation: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        processed_query = self.query_processor.process(
            query,
            conversation=conversation,
        )
        query_embedding = self.mm_client.embed_query(processed_query)

        multimodal_text_results = self.db_client.search_by_embedding(
            embedding=query_embedding,
            collection_name=self.text_collection_name,
            k=text_k,
        )
        image_results = self.db_client.search_by_embedding(
            embedding=query_embedding,
            collection_name=self.image_collection_name,
            k=image_k,
        )

        ocr_text_results: List[SearchResult] = []
        if self.ocr_collection_name and self.text_embedding_client:
            try:
                ocr_query_embedding = self.text_embedding_client.embed_query(
                    processed_query
                )
                ocr_text_results = self.db_client.search_by_embedding(
                    embedding=ocr_query_embedding,
                    collection_name=self.ocr_collection_name,
                    k=ocr_k,
                )
            except Exception as exc:
                # OCR 是补充召回通道，失败时保留多模态主链路的可用性。
                logger.warning("OCR 文本检索失败，已降级为多模态检索: %s", exc)

        text_results = self._fuse_text_results(
            multimodal_text_results,
            ocr_text_results,
            limit=text_k,
        )

        return {
            "query": processed_query,
            "text_results": text_results,
            "multimodal_text_results": multimodal_text_results,
            "ocr_text_results": ocr_text_results,
            "image_results": image_results,
        }

    @staticmethod
    def _result_key(content: str, metadata: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            metadata.get("source_file", ""),
            metadata.get("page", ""),
            metadata.get("chunk_index", metadata.get("block_index", "")),
            content.strip(),
        )

    @classmethod
    def _fuse_text_results(
        cls,
        multimodal_results: List[SearchResult],
        ocr_results: List[SearchResult],
        limit: int,
        rrf_k: int = 60,
    ) -> List[SearchResult]:
        """使用 RRF 融合不同 Embedding 空间的排序，避免直接比较距离分数。"""
        fused: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for channel, results in (
            ("multimodal_text", multimodal_results),
            ("ocr_text", ocr_results),
        ):
            for rank, (content, metadata, channel_score) in enumerate(results, 1):
                key = cls._result_key(content, metadata)
                if key not in fused:
                    fused[key] = {
                        "content": content,
                        "metadata": dict(metadata or {}),
                        "score": 0.0,
                        "channels": [],
                    }
                item = fused[key]
                item["score"] += 1.0 / (rrf_k + rank)
                item["channels"].append(channel)
                item["metadata"][f"{channel}_score"] = round(channel_score, 6)

        ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
        output: List[SearchResult] = []
        for item in ranked[:limit]:
            item["metadata"]["retrieval_channels"] = ",".join(item["channels"])
            output.append((item["content"], item["metadata"], item["score"]))
        return output

    def search(
        self,
        query: str,
        k: int = 8,
        image_k: int = 4,
        conversation: Optional[List[Dict[str, Any]]] = None,
    ) -> List[SearchResult]:
        payload = self.retrieve(
            query=query,
            text_k=k,
            image_k=image_k,
            conversation=conversation,
        )
        return payload["text_results"]

    @staticmethod
    def _parse_bbox(raw_bbox: Any) -> Dict[str, Any]:
        if isinstance(raw_bbox, dict):
            return raw_bbox
        if isinstance(raw_bbox, str):
            try:
                return json.loads(raw_bbox)
            except Exception:
                return {}
        return {}

    def format_context(
        self,
        results: List[SearchResult],
        max_context_length: int = 2000,
        include_metadata: bool = True,
    ) -> str:
        return self.build_context(
            results,
            max_context_length=max_context_length,
            include_metadata=include_metadata,
        ).text

    def build_context(
        self,
        results: List[SearchResult],
        max_context_length: int = 2000,
        include_metadata: bool = True,
    ) -> ContextBuildResult:
        """构建教材上下文，并保留与引用编号一致的实际入选结果。"""
        return self.context_builder.build(
            results=results,
            max_context_length=max_context_length,
            include_metadata=include_metadata,
            format_type="textbook",
        )

    def format_image_context(
        self,
        image_results: List[SearchResult],
        max_items: int = 3,
    ) -> str:
        if not image_results:
            return ""

        lines = []
        for idx, (_, metadata, _) in enumerate(image_results[:max_items], 1):
            page = metadata.get("page", "")
            source_file = metadata.get("source_file", "")
            nearby_text = metadata.get("nearby_text", "").strip()
            lines.append(
                f"[图{idx}] 来源={source_file} 第{page}页\n"
                f"相关文字：{nearby_text or '无'}"
            )
        return "\n\n".join(lines)

    def serialize_images(
        self,
        image_results: List[SearchResult],
    ) -> List[Dict[str, Any]]:
        images = []
        for idx, (_, metadata, score) in enumerate(image_results, 1):
            images.append(
                {
                    "image_ref": f"图{idx}",
                    "image_url": metadata.get("image_url"),
                    "image_path": metadata.get("image_path"),
                    "source_file": metadata.get("source_file"),
                    "page": metadata.get("page"),
                    "bbox": self._parse_bbox(metadata.get("bbox")),
                    "nearby_text": metadata.get("nearby_text", ""),
                    "score": round(score, 4),
                }
            )
        return images

    def get_stats(self) -> Dict[str, Any]:
        text_stats = self.db_client.get_stats(collection_name=self.text_collection_name)
        image_stats = self.db_client.get_stats(
            collection_name=self.image_collection_name
        )
        ocr_stats = (
            self.db_client.get_stats(collection_name=self.ocr_collection_name)
            if self.ocr_collection_name
            else None
        )
        return {
            "connected": (
                text_stats.get("connected")
                and image_stats.get("connected")
                and (ocr_stats is None or ocr_stats.get("connected"))
            ),
            "text_collection": self.text_collection_name,
            "image_collection": self.image_collection_name,
            "ocr_collection": self.ocr_collection_name,
            "text_records": text_stats.get("total_records", 0),
            "image_records": image_stats.get("total_records", 0),
            "ocr_records": ocr_stats.get("total_records", 0) if ocr_stats else 0,
        }

    def get_sources(
        self,
        results: List[SearchResult],
        reply: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if reply:
            cited_indices = set(int(m) for m in RE_CITE.findall(reply))
        else:
            cited_indices = set(range(1, len(results) + 1))

        seen = set()
        sources = []

        for i, (_, meta, _) in enumerate(results, 1):
            if i not in cited_indices:
                continue

            source_file = meta.get("source_file", "")
            page = meta.get("page", "")
            key = (source_file, page)
            if key in seen:
                continue
            seen.add(key)

            sources.append(
                {
                    "source_file": source_file,
                    "chapter": meta.get("chapter", ""),
                    "section": meta.get("section", ""),
                    "page": page,
                }
            )

            if len(sources) >= 5:
                break

        return sources
