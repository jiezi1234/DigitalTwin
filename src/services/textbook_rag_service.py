"""
教材 RAG 服务（多模态版本）
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.db_client import DBClient
from src.infrastructure.text_embedding_client import TextEmbeddingClient
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.multimodal_embedding_client import MultiModalEmbeddingClient
from src.rag.query_processor import QueryProcessor

logger = logging.getLogger(__name__)

RE_CITE = re.compile(r'\[(\d+)\]')
RE_ZH = re.compile(r'[\u4e00-\u9fff]')
RE_TOKEN = re.compile(r"[a-zA-Z0-9%]+|[\u4e00-\u9fff]+")
RE_CHAPTER_ITEM = re.compile(r"\bchapter\s*\d+\b", re.IGNORECASE)

SearchResult = Tuple[str, Dict[str, Any], float]


class TextbookRAGService:
    """教材 RAG 服务（文本块 + 图片双路召回 + OCR 托底）"""

    def __init__(
        self,
        llm_client: LLMClient,
        db_client: DBClient,
        text_collection_name: str = "textbook_mm_text_embeddings",
        image_collection_name: str = "textbook_mm_image_embeddings",
        ocr_collection_name: Optional[str] = "textbook_ocr_text_embeddings",
        enable_query_rewriting: bool = True,
        mm_client: Optional[MultiModalEmbeddingClient] = None,
        text_client: Optional[TextEmbeddingClient] = None,
    ):
        self.llm_client = llm_client
        self.db_client = db_client
        self.text_collection_name = text_collection_name
        self.image_collection_name = image_collection_name
        self.ocr_collection_name = ocr_collection_name
        self.mm_client = mm_client or MultiModalEmbeddingClient()
        self.text_client = text_client or TextEmbeddingClient()
        self.query_processor = QueryProcessor(
            llm_client=llm_client,
            enable_coreference_resolution=False,
            enable_query_rewriting=enable_query_rewriting,
        )

    @staticmethod
    def _merge_alternating(
        primary: List[SearchResult],
        secondary: List[SearchResult],
        limit: int,
    ) -> List[SearchResult]:
        """交替合并两个结果流，保证双集合都有机会进入最终候选。"""
        merged: List[SearchResult] = []
        i = j = 0
        while len(merged) < limit and (i < len(primary) or j < len(secondary)):
            if i < len(primary):
                merged.append(primary[i])
                i += 1
                if len(merged) >= limit:
                    break
            if j < len(secondary):
                merged.append(secondary[j])
                j += 1
        return merged[:limit]

    def _expand_crosslingual_query(self, query: str) -> str:
        """
        对中文查询补充英文检索关键词，提升英文课件的召回率。
        """
        if not query or not RE_ZH.search(query):
            return query

        prompt = (
            "你是检索改写器。将下面中文问题补充为英文检索关键词，用逗号分隔，"
            "只输出关键词，不要解释。\n"
            f"问题：{query}"
        )
        try:
            expanded_en = self.llm_client.call(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=80,
            )
            if expanded_en:
                expanded_en = expanded_en.strip()
                if expanded_en:
                    return f"{query}，{expanded_en}"
        except Exception as e:
            logger.warning(f"Cross-lingual query expansion failed: {e}")
        return query

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        tokens = [t.lower() for t in RE_TOKEN.findall(text or "")]
        return [t for t in tokens if len(t) >= 2]

    @staticmethod
    def _keyword_score(content: str, keywords: List[str]) -> int:
        if not keywords:
            return 0
        content_l = (content or "").lower()
        return sum(1 for kw in keywords if kw in content_l)

    @staticmethod
    def _assessment_pattern_bonus(content: str, query: str) -> int:
        """
        对“成绩构成/考核比例”这类问题给强特征加权：
        命中 assessment/score/grade/exam 且包含百分比时显著加分。
        """
        q = query or ""
        if not any(k in q for k in ["成绩", "构成", "考核", "评分", "分数", "grade", "score", "assessment"]):
            return 0
        c = (content or "").lower()
        has_assessment_word = any(t in c for t in ["assessment", "score", "grade", "exam", "exercise", "lab"])
        has_percent = "%" in c or "percent" in c
        if has_assessment_word and has_percent:
            return 100
        if has_assessment_word:
            return 20
        return 0

    @staticmethod
    def _chapter_overview_bonus(content: str, query: str) -> int:
        """
        对“课程章节/有哪些章节”问题优先召回“章节总览页”：
        - 若同一段落命中多个 Chapter N，说明更可能是目录/总览。
        """
        q = query or ""
        is_chapter_query = any(k in q for k in ["章节", "chapter", "目录", "课程结构"])
        if not is_chapter_query:
            return 0
        c = content or ""
        chapter_hits = len(RE_CHAPTER_ITEM.findall(c))
        if chapter_hits >= 5:
            return 140 + chapter_hits * 5
        if chapter_hits >= 2:
            return 80 + chapter_hits * 5
        if chapter_hits == 1:
            return 15
        return 0

    @staticmethod
    def _domain_hint_tokens(query: str) -> List[str]:
        q = query or ""
        hints: List[str] = []
        if any(k in q for k in ["成绩", "构成", "考核", "评分", "分数", "grade", "score"]):
            hints.extend([
                "assessment", "score", "grade", "exam", "final exam",
                "middle exam", "exercise", "lab", "10%", "20%", "60%",
            ])
        if any(k in q for k in ["章节", "chapter", "目录", "课程结构"]):
            hints.extend([
                "chapter 1", "chapter 2", "chapter 3", "chapter 4",
                "chapter 5", "chapter 6", "chapter 7", "chapter 8",
                "chapter 9", "chapter 10", "chapter 11", "chapter 12",
                "so what is it all about",
            ])
        return hints

    def _rerank_results(
        self,
        results: List[SearchResult],
        query: str,
        top_k: int,
    ) -> List[SearchResult]:
        if not results:
            return results
        keywords = self._extract_keywords(query)
        keywords.extend(self._domain_hint_tokens(query))
        if not keywords:
            return results[:top_k]

        scored = []
        for idx, item in enumerate(results):
            content, metadata, sim = item
            kscore = self._keyword_score(content, keywords)
            bonus = self._assessment_pattern_bonus(content, query)
            bonus += self._chapter_overview_bonus(content, query)
            # 关键词优先，其次向量相似度，再保持原顺序稳定
            scored.append((kscore + bonus, float(sim), -idx, item))

        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        reranked = [x[3] for x in scored]
        return reranked[:top_k]

    def retrieve(
        self,
        query: str,
        text_k: int = 8,
        image_k: int = 4,
    ) -> Dict[str, Any]:
        processed_query = self.query_processor.process(query)
        processed_query = self._expand_crosslingual_query(processed_query)
        logger.info(f"[Tutor Retrieve] query='{query}' processed='{processed_query}'")
        mm_query_embedding = self.mm_client.embed_query(processed_query)
        text_query_embedding = self.text_client.embed_texts([processed_query])[0]
        fetch_k = max(text_k * 6, 40)
        notes_fetch_k = fetch_k
        textbook_fetch_k = fetch_k
        keywords = self._extract_keywords(processed_query)
        keywords.extend(self._domain_hint_tokens(processed_query))
        keyword_fetch_k = max(text_k * 4, 24)

        try:
            mm_text_results = self.db_client.search_by_embedding(
                embedding=mm_query_embedding,
                collection_name=self.text_collection_name,
                k=fetch_k,
            )
        except Exception as e:
            logger.warning(f"Failed to query mm_text collection: {e}")
            mm_text_results = []
            
        try:
            ocr_text_results = []
            if self.ocr_collection_name:
                ocr_text_results = self.db_client.search_by_embedding(
                    embedding=text_query_embedding,
                    collection_name=self.ocr_collection_name,
                    k=textbook_fetch_k,
                )
        except Exception as e:
            logger.warning(f"Failed to query ocr_text collection: {e}")
            ocr_text_results = []

        try:
            lexical_text_results = self.db_client.search_by_keywords(
                collection_name=self.text_collection_name,
                keywords=keywords,
                k=keyword_fetch_k,
            )
        except Exception as e:
            logger.warning(f"Failed to keyword-search mm_text collection: {e}")
            lexical_text_results = []

        try:
            lexical_ocr_results = []
            if self.ocr_collection_name:
                lexical_ocr_results = self.db_client.search_by_keywords(
                    collection_name=self.ocr_collection_name,
                    keywords=keywords,
                    k=keyword_fetch_k,
                )
        except Exception as e:
            logger.warning(f"Failed to keyword-search ocr_text collection: {e}")
            lexical_ocr_results = []

        try:
            if image_k > 0:
                image_results = self.db_client.search_by_embedding(
                    embedding=mm_query_embedding,
                    collection_name=self.image_collection_name,
                    k=image_k,
                )
            else:
                image_results = []
        except Exception as e:
            logger.warning(f"Failed to query mm_image collection: {e}")
            image_results = []

        # Interleave mm_text_results and ocr_text_results instead of pure score sorting,
        # because distances from qwen3-vl-embedding and text-embedding-v3 have different scales.
        # notes collection + textbook collection 分开重排，再按配额合并
        notes_pool = mm_text_results + lexical_text_results
        textbook_pool = ocr_text_results + lexical_ocr_results

        notes_reranked = self._rerank_results(
            notes_pool,
            query=processed_query,
            top_k=max(text_k * 3, 20),
        )
        textbook_reranked = self._rerank_results(
            textbook_pool,
            query=processed_query,
            top_k=max(text_k * 3, 20),
        )

        # 默认各占一半；若某一侧为空，则另一侧补齐。
        notes_quota = text_k
        textbook_quota = text_k
        if notes_reranked and textbook_reranked:
            notes_quota = max(1, text_k // 2)
            textbook_quota = max(1, text_k - notes_quota)
        elif notes_reranked:
            notes_quota = text_k * 2
            textbook_quota = 0
        elif textbook_reranked:
            notes_quota = 0
            textbook_quota = text_k * 2

        notes_selected = notes_reranked[:notes_quota]
        textbook_selected = textbook_reranked[:textbook_quota]
        combined_text_results = self._merge_alternating(
            primary=notes_selected,
            secondary=textbook_selected,
            limit=text_k * 2,
        )
        logger.info(
            "[Tutor Retrieve] notes_selected=%s textbook_selected=%s merged=%s",
            len(notes_selected),
            len(textbook_selected),
            len(combined_text_results),
        )
        
        # Deduplicate
        seen_content = set()
        unique_text_results = []
        for res in combined_text_results:
            content = res[0]
            if content not in seen_content:
                seen_content.add(content)
                unique_text_results.append(res)

        # 注意：这里不能在重排前截断，否则后追加的 lexical 结果会被完全丢弃。
        # 先全量去重，再重排，再截断。

        unique_text_results = self._rerank_results(
            unique_text_results,
            query=processed_query,
            top_k=text_k * 2,
        )

        return {
            "query": processed_query,
            "text_results": unique_text_results,
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
        
        ocr_stats = {}
        if self.ocr_collection_name:
            ocr_stats = self.db_client.get_stats(collection_name=self.ocr_collection_name)

        return {
            "connected": text_stats.get("connected") and image_stats.get("connected"),
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
    ) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
        if reply:
            cited_indices = set(int(m) for m in RE_CITE.findall(reply))
        else:
            cited_indices = set(range(1, len(results) + 1))

        seen = {}
        sources = []
        index_mapping = {}

        for i, (_, meta, _) in enumerate(results, 1):
            if i not in cited_indices and reply is not None:
                continue

            source_file = meta.get("source_file", "")
            page = meta.get("page", "")
            key = (source_file, page)
            
            if key in seen:
                index_mapping[i] = seen[key]
                continue

            if len(sources) < 5:
                sources.append({
                    "source_file": source_file,
                    "chapter": meta.get("chapter", ""),
                    "section": meta.get("section", ""),
                    "page": page,
                })
                new_idx = len(sources)
                seen[key] = new_idx
                index_mapping[i] = new_idx

        return sources, index_mapping
