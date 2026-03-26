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
from src.rag.query_processor import QueryProcessor

logger = logging.getLogger(__name__)

RE_CITE = re.compile(r'\[(\d+)\]')

SearchResult = Tuple[str, Dict[str, Any], float]


class TextbookRAGService:
    """教材 RAG 服务（文本块 + 图片双路召回）"""

    def __init__(
        self,
        llm_client: LLMClient,
        db_client: DBClient,
        text_collection_name: str = "textbook_mm_text_embeddings",
        image_collection_name: str = "textbook_mm_image_embeddings",
        enable_query_rewriting: bool = True,
        mm_client: Optional[MultiModalEmbeddingClient] = None,
    ):
        self.llm_client = llm_client
        self.db_client = db_client
        self.text_collection_name = text_collection_name
        self.image_collection_name = image_collection_name
        self.mm_client = mm_client or MultiModalEmbeddingClient()
        self.query_processor = QueryProcessor(
            llm_client=llm_client,
            enable_coreference_resolution=False,
            enable_query_rewriting=enable_query_rewriting,
        )

    def retrieve(
        self,
        query: str,
        text_k: int = 8,
        image_k: int = 4,
    ) -> Dict[str, Any]:
        processed_query = self.query_processor.process(query)
        query_embedding = self.mm_client.embed_query(processed_query)

        text_results = self.db_client.search_by_embedding(
            embedding=query_embedding,
            collection_name=self.text_collection_name,
            k=text_k,
        )
        image_results = self.db_client.search_by_embedding(
            embedding=query_embedding,
            collection_name=self.image_collection_name,
            k=image_k,
        )

        return {
            "query": processed_query,
            "text_results": text_results,
            "image_results": image_results,
        }

    def search(
        self,
        query: str,
        k: int = 8,
        image_k: int = 4,
    ) -> List[SearchResult]:
        payload = self.retrieve(query=query, text_k=k, image_k=image_k)
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
        if not results:
            return ""

        lines: List[str] = []
        total_length = 0

        for idx, (content, metadata, _) in enumerate(results, 1):
            source_file = metadata.get("source_file", "")
            page = metadata.get("page", "")
            location_parts = [part for part in [source_file, f"第{page}页" if page else ""] if part]
            location = " > ".join(location_parts)
            record = f"[{idx}]【{location}】\n{content.strip()}\n" if include_metadata else f"[{idx}] {content.strip()}"

            if total_length + len(record) > max_context_length:
                break
            lines.append(record)
            total_length += len(record)

        return "\n".join(lines)

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
            images.append({
                "image_ref": f"图{idx}",
                "image_url": metadata.get("image_url"),
                "image_path": metadata.get("image_path"),
                "source_file": metadata.get("source_file"),
                "page": metadata.get("page"),
                "bbox": self._parse_bbox(metadata.get("bbox")),
                "nearby_text": metadata.get("nearby_text", ""),
                "score": round(score, 4),
            })
        return images

    def get_stats(self) -> Dict[str, Any]:
        text_stats = self.db_client.get_stats(collection_name=self.text_collection_name)
        image_stats = self.db_client.get_stats(collection_name=self.image_collection_name)
        return {
            "connected": text_stats.get("connected") and image_stats.get("connected"),
            "text_collection": self.text_collection_name,
            "image_collection": self.image_collection_name,
            "text_records": text_stats.get("total_records", 0),
            "image_records": image_stats.get("total_records", 0),
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

            sources.append({
                "source_file": source_file,
                "chapter": meta.get("chapter", ""),
                "section": meta.get("section", ""),
                "page": page,
            })

            if len(sources) >= 5:
                break

        return sources
