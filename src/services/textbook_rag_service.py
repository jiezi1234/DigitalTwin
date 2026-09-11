"""
教材 RAG 服务（多模态版本）
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.db_client import DBClient
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.multimodal_embedding_client import MultiModalEmbeddingClient
from src.infrastructure.text_embedding_client import TextEmbeddingClient
from src.rag.query_processor import QueryProcessor
from src.rag.context_builder import ContextBuilder, ContextBuildResult
from src.rag.citation_validator import (
    CitationGroundingValidation,
    CitationGroundingValidator,
    CitationValidation,
    CitationValidator,
)
from src.rag.evidence_policy import EvidenceAssessment, EvidenceConfidencePolicy
from src.rag.bm25_retriever import BM25Retriever
from src.rag.llm_reranker import LLMReranker

logger = logging.getLogger(__name__)

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
        citation_validator: Optional[CitationValidator] = None,
        evidence_policy: Optional[EvidenceConfidencePolicy] = None,
        min_text_evidence_score: float = 0.45,
        min_image_evidence_score: float = 0.45,
        min_evidence_items: int = 1,
        lexical_retriever: Optional[BM25Retriever] = None,
        reranker: Optional[LLMReranker] = None,
        enable_hybrid_search: bool = False,
        bm25_candidates: int = 30,
        rrf_k: int = 60,
        mm_text_weight: float = 1.0,
        ocr_text_weight: float = 0.9,
        bm25_weight: float = 0.8,
        enable_reranking: bool = False,
        rerank_candidates: int = 20,
        citation_grounding_validator: Optional[CitationGroundingValidator] = None,
        min_citation_support_score: float = 0.45,
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
        self.citation_validator = citation_validator or CitationValidator()
        self.evidence_policy = evidence_policy or EvidenceConfidencePolicy(
            min_text_score=min_text_evidence_score,
            min_image_score=min_image_evidence_score,
            min_items=min_evidence_items,
        )
        self.enable_hybrid_search = enable_hybrid_search
        self.lexical_retriever = lexical_retriever or (
            BM25Retriever(db_client) if enable_hybrid_search else None
        )
        self.enable_reranking = enable_reranking
        self.reranker = reranker or (
            LLMReranker(llm_client) if enable_reranking else None
        )
        self.bm25_candidates = max(1, bm25_candidates)
        self.rrf_k = max(1, rrf_k)
        self.mm_text_weight = max(0.0, mm_text_weight)
        self.ocr_text_weight = max(0.0, ocr_text_weight)
        self.bm25_weight = max(0.0, bm25_weight)
        self.rerank_candidates = max(1, rerank_candidates)
        self.citation_grounding_validator = (
            citation_grounding_validator
            or CitationGroundingValidator(min_citation_support_score)
        )
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
        candidate_k = max(
            text_k,
            self.bm25_candidates if self.enable_hybrid_search else text_k,
            self.rerank_candidates if self.enable_reranking else text_k,
        )
        query_embedding = self.mm_client.embed_query(processed_query)

        multimodal_text_results = self.db_client.search_by_embedding(
            embedding=query_embedding,
            collection_name=self.text_collection_name,
            k=candidate_k,
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
                    k=max(ocr_k, candidate_k),
                )
            except Exception as exc:
                # OCR 是补充召回通道，失败时保留多模态主链路的可用性。
                logger.warning("OCR 文本检索失败，已降级为多模态检索: %s", exc)

        bm25_results: List[SearchResult] = []
        if self.enable_hybrid_search and self.lexical_retriever:
            try:
                bm25_results = self.lexical_retriever.search(
                    query=processed_query,
                    collection_name=self.text_collection_name,
                    k=self.bm25_candidates,
                )
            except Exception as exc:
                logger.warning("教材 BM25 检索失败，已降级为向量召回: %s", exc)

        text_results = self._fuse_text_results(
            multimodal_text_results,
            ocr_text_results,
            bm25_results=bm25_results,
            limit=max(text_k, self.rerank_candidates),
            rrf_k=self.rrf_k,
            mm_text_weight=self.mm_text_weight,
            ocr_text_weight=self.ocr_text_weight,
            bm25_weight=self.bm25_weight,
        )
        if self.enable_reranking and self.reranker:
            text_results = self.reranker.rerank(
                query=processed_query,
                results=text_results,
                top_k=text_k,
                candidate_limit=self.rerank_candidates,
            )
        else:
            text_results = text_results[:text_k]

        return {
            "query": processed_query,
            "text_results": text_results,
            "multimodal_text_results": multimodal_text_results,
            "ocr_text_results": ocr_text_results,
            "bm25_results": bm25_results,
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
        bm25_results: Optional[List[SearchResult]] = None,
        mm_text_weight: float = 1.0,
        ocr_text_weight: float = 0.9,
        bm25_weight: float = 0.8,
    ) -> List[SearchResult]:
        """使用加权 RRF 融合多模态、OCR 与 BM25 的不可比分数。"""
        fused: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for channel, weight, results in (
            ("multimodal_text", max(0.0, mm_text_weight), multimodal_results),
            ("ocr_text", max(0.0, ocr_text_weight), ocr_results),
            ("bm25", max(0.0, bm25_weight), bm25_results or []),
        ):
            if weight == 0:
                continue
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
                item["score"] += weight / (max(1, rrf_k) + rank)
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
            cited_indices = set(self.validate_citations(reply, results).valid_indices)
        else:
            cited_indices = set(range(1, len(results) + 1))

        sources = []

        for i, (_, meta, _) in enumerate(results, 1):
            if i not in cited_indices:
                continue

            sources.append(
                {
                    "citation_index": i,
                    "source_file": meta.get("source_file", ""),
                    "chapter": meta.get("chapter", ""),
                    "section": meta.get("section", ""),
                    "page": meta.get("page", ""),
                }
            )

            if len(sources) >= 8:
                break

        return sources

    def validate_citations(
        self,
        reply: Optional[str],
        results: List[SearchResult],
    ) -> CitationValidation:
        """校验文本引用是否指向实际进入模型上下文的片段。"""
        return self.citation_validator.validate(reply, context_count=len(results))

    def validate_citation_grounding(
        self,
        reply: Optional[str],
        results: List[SearchResult],
    ) -> CitationGroundingValidation:
        """检查回答事实句的引用覆盖率及引用内容的词汇支持度。"""
        return self.citation_grounding_validator.validate(reply, results)

    def assess_evidence(
        self, text_results: List[SearchResult], image_results: List[Any]
    ) -> EvidenceAssessment:
        """根据各检索通道的原始相似度评估证据是否足以支撑回答。"""
        return self.evidence_policy.assess(text_results, image_results)

    def has_evidence(
        self, text_results: List[SearchResult], image_results: List[Any]
    ) -> bool:
        """兼容布尔接口；新代码应优先使用 ``assess_evidence``。"""
        return self.assess_evidence(text_results, image_results).sufficient
