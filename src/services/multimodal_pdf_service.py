"""
PDF 多模态索引构建服务
"""

import glob
import json
import logging
import os
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.infrastructure.document import Document
from src.infrastructure.multimodal_embedding_client import MultiModalEmbeddingClient
from src.loaders.base import DataLoader
from src.loaders.pdf_loader import PDFLoader

logger = logging.getLogger(__name__)


class MultiModalPDFIndexService:
    """将 PDF 的文本块和图片写入两个多模态 collection"""

    def __init__(self, db_client, embedding_client: MultiModalEmbeddingClient):
        self.db_client = db_client
        self.embedding_client = embedding_client

    @staticmethod
    def _read_structured_json(json_path: str) -> Dict[str, Any]:
        with open(json_path, "r", encoding="utf-8") as json_file:
            return json.load(json_file)

    @staticmethod
    def _load_tracking(tracking_file: Optional[str]) -> Dict[str, Any]:
        if not tracking_file or not os.path.exists(tracking_file):
            return {"completed_text_ids": [], "completed_image_ids": []}
        with open(tracking_file, "r", encoding="utf-8") as tracking_handle:
            return json.load(tracking_handle)

    @staticmethod
    def _save_tracking(tracking_file: Optional[str], tracking_data: Dict[str, Any]) -> None:
        if not tracking_file:
            return
        os.makedirs(os.path.dirname(tracking_file) or ".", exist_ok=True)
        with open(tracking_file, "w", encoding="utf-8") as tracking_handle:
            json.dump(tracking_data, tracking_handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _resolve_image_url(relative_path: str) -> str:
        normalized = relative_path.replace(os.sep, "/").lstrip("/")
        return f"/exports/{normalized}"

    @staticmethod
    def _get_nearby_text(
        text_blocks: Sequence[Dict[str, Any]],
        anchor_index: int,
        window: int = 1,
    ) -> str:
        if not text_blocks:
            return ""
        start = max(0, anchor_index - window)
        end = min(len(text_blocks), anchor_index + window + 1)
        return "\n".join(
            block.get("content", "")
            for block in text_blocks[start:end]
            if block.get("content")
        ).strip()

    def build_text_documents(self, structured_data: Dict[str, Any]) -> List[Document]:
        source_file = structured_data["source_file"]
        documents: List[Document] = []

        for page_data in structured_data.get("pages", []):
            for text_block in page_data.get("text_blocks", []):
                metadata = {
                    "source": "pdf_text_block",
                    "source_file": source_file,
                    "page": page_data["page"],
                    "block_index": text_block["block_index"],
                    "bbox": json.dumps(text_block["bbox"], ensure_ascii=False),
                    "content_type": "text_block",
                }
                doc_id = DataLoader.generate_hash(text_block["content"], metadata)
                documents.append(
                    Document(
                        content=text_block["content"],
                        metadata=metadata,
                        doc_id=doc_id,
                    )
                )
        return documents

    def build_image_documents(
        self,
        structured_data: Dict[str, Any],
        export_dir: str,
    ) -> Tuple[List[Document], List[str]]:
        source_file = structured_data["source_file"]
        export_subdir = os.path.basename(export_dir.rstrip(os.sep))
        documents: List[Document] = []
        image_paths: List[str] = []
        deduped_images: Dict[str, Dict[str, Any]] = {}

        for page_data in structured_data.get("pages", []):
            text_blocks = page_data.get("text_blocks", [])
            for image_entry in page_data.get("images", []):
                relative_path = image_entry["file"]
                stored_path = os.path.join(export_subdir, relative_path)
                image_path = os.path.join(export_dir, relative_path)
                nearby_text = self._get_nearby_text(text_blocks, image_entry["image_index"])
                image_hash = image_entry.get("image_hash") or stored_path

                if image_hash not in deduped_images:
                    deduped_images[image_hash] = {
                        "display_text": f"[image] {source_file} 第{page_data['page']}页",
                        "metadata": {
                            "source": "pdf_image",
                            "source_file": source_file,
                            "page": page_data["page"],
                            "image_index": image_entry["image_index"],
                            "bbox": json.dumps(image_entry["bbox"], ensure_ascii=False),
                            "image_hash": image_hash,
                            "image_path": stored_path,
                            "image_url": self._resolve_image_url(stored_path),
                            "extension": image_entry.get("extension"),
                            "width": image_entry.get("width"),
                            "height": image_entry.get("height"),
                            "nearby_text": nearby_text,
                            "content_type": "image",
                            "occurrence_pages": [page_data["page"]],
                            "occurrence_count": 1,
                        },
                        "image_path": image_path,
                    }
                    continue

                existing_meta = deduped_images[image_hash]["metadata"]
                existing_meta["occurrence_count"] += 1
                if page_data["page"] not in existing_meta["occurrence_pages"]:
                    existing_meta["occurrence_pages"].append(page_data["page"])
                if nearby_text and nearby_text not in existing_meta["nearby_text"]:
                    merged = "\n".join(
                        part for part in [existing_meta["nearby_text"], nearby_text] if part
                    )
                    existing_meta["nearby_text"] = merged[:1000]

        for item in deduped_images.values():
            metadata = item["metadata"]
            metadata["occurrence_pages"] = json.dumps(metadata["occurrence_pages"], ensure_ascii=False)
            doc_id = DataLoader.generate_hash(item["display_text"], metadata)
            documents.append(
                Document(
                    content=item["display_text"],
                    metadata=metadata,
                    doc_id=doc_id,
                )
            )
            image_paths.append(item["image_path"])

        return documents, image_paths

    def _bulk_import_documents(
        self,
        collection_name: str,
        documents: List[Document],
        embeddings: List[List[float]],
    ) -> int:
        if not documents:
            return 0

        flat_docs = [doc.content for doc in documents]
        flat_metas = [doc.metadata for doc in documents]
        flat_ids = [doc.doc_id for doc in documents]
        return self.db_client.bulk_import(
            collection_name=collection_name,
            documents=flat_docs,
            embeddings=embeddings,
            metadatas=flat_metas,
            ids=flat_ids,
        )

    @staticmethod
    def _batched(items: Sequence[Any], batch_size: int) -> List[Sequence[Any]]:
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    def _embed_text_documents(
        self,
        documents: List[Document],
        batch_size: int,
        max_workers: int,
    ) -> List[List[float]]:
        if not documents:
            return []

        batches = self._batched(documents, batch_size)
        results: List[Optional[List[List[float]]]] = [None] * len(batches)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self.embedding_client.embed_texts,
                    [doc.content for doc in batch],
                ): idx
                for idx, batch in enumerate(batches)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        embeddings: List[List[float]] = []
        for batch_vectors in results:
            if batch_vectors:
                embeddings.extend(batch_vectors)
        return embeddings

    def _import_text_batches(
        self,
        collection_name: str,
        documents: List[Document],
        batch_size: int,
        max_workers: int,
        tracking_file: Optional[str] = None,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> Tuple[int, int]:
        tracking_data = self._load_tracking(tracking_file)
        completed_ids = set(tracking_data.get("completed_text_ids", []))
        pending_documents = [doc for doc in documents if doc.doc_id not in completed_ids]

        if not pending_documents:
            if progress_callback:
                progress_callback(stage="text", event="skipped_all", total=len(documents), pending=0)
            return 0, len(documents)

        batches = self._batched(pending_documents, batch_size)
        imported_count = 0
        if progress_callback:
            progress_callback(
                stage="text",
                event="start",
                total=len(documents),
                pending=len(pending_documents),
                total_batches=len(batches),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(
                    self.embedding_client.embed_texts,
                    [doc.content for doc in batch],
                ): list(batch)
                for batch in batches
            }
            for future in concurrent.futures.as_completed(future_to_batch):
                batch = future_to_batch[future]
                embeddings = future.result()
                imported_count += self._bulk_import_documents(collection_name, batch, embeddings)
                completed_ids.update(doc.doc_id for doc in batch)
                tracking_data["completed_text_ids"] = sorted(completed_ids)
                self._save_tracking(tracking_file, tracking_data)
                if progress_callback:
                    progress_callback(
                        stage="text",
                        event="batch_done",
                        batch_size=len(batch),
                        imported_count=imported_count,
                    )

        return imported_count, len(documents) - len(pending_documents)

    def _embed_images(
        self,
        image_paths: List[str],
        batch_size: int,
        max_workers: int,
    ) -> List[List[float]]:
        if not image_paths:
            return []

        batches = self._batched(image_paths, batch_size)
        results: List[Optional[List[List[float]]]] = [None] * len(batches)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.embedding_client.embed_images, list(batch)): idx
                for idx, batch in enumerate(batches)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        embeddings: List[List[float]] = []
        for batch_vectors in results:
            if batch_vectors:
                embeddings.extend(batch_vectors)
        return embeddings

    def _import_image_batches(
        self,
        collection_name: str,
        documents: List[Document],
        image_paths: List[str],
        batch_size: int,
        max_workers: int,
        tracking_file: Optional[str] = None,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> Tuple[int, int]:
        tracking_data = self._load_tracking(tracking_file)
        completed_ids = set(tracking_data.get("completed_image_ids", []))
        pending_pairs = [
            (doc, path)
            for doc, path in zip(documents, image_paths)
            if doc.doc_id not in completed_ids
        ]

        if not pending_pairs:
            if progress_callback:
                progress_callback(stage="image", event="skipped_all", total=len(documents), pending=0)
            return 0, len(documents)

        batches = self._batched(pending_pairs, batch_size)
        imported_count = 0
        if progress_callback:
            progress_callback(
                stage="image",
                event="start",
                total=len(documents),
                pending=len(pending_pairs),
                total_batches=len(batches),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(
                    self.embedding_client.embed_images,
                    [path for _, path in batch],
                ): list(batch)
                for batch in batches
            }
            for future in concurrent.futures.as_completed(future_to_batch):
                batch = future_to_batch[future]
                embeddings = future.result()
                batch_docs = [doc for doc, _ in batch]
                imported_count += self._bulk_import_documents(collection_name, batch_docs, embeddings)
                completed_ids.update(doc.doc_id for doc, _ in batch)
                tracking_data["completed_image_ids"] = sorted(completed_ids)
                self._save_tracking(tracking_file, tracking_data)
                if progress_callback:
                    progress_callback(
                        stage="image",
                        event="batch_done",
                        batch_size=len(batch_docs),
                        imported_count=imported_count,
                    )

        return imported_count, len(documents) - len(pending_pairs)

    def index_pdf(
        self,
        pdf_path: str,
        output_root: str,
        text_collection: str,
        image_collection: str,
        text_batch_size: int = 8,
        text_embed_workers: int = 4,
        image_batch_size: int = 1,
        image_embed_workers: int = 2,
        tracking_file: Optional[str] = None,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> Dict[str, Any]:
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        export_dir = os.path.join(output_root, pdf_name)

        loader = PDFLoader(filepath=pdf_path, ocr_enabled=False, max_workers=1)
        export_data = loader.export_structured(output_dir=export_dir)
        structured_data = self._read_structured_json(export_data["json_file"])

        text_docs = self.build_text_documents(structured_data)
        image_docs, image_paths = self.build_image_documents(structured_data, export_dir)
        if progress_callback:
            progress_callback(
                stage="prepare",
                event="ready",
                pdf_path=pdf_path,
                text_total=len(text_docs),
                image_total=len(image_docs),
            )

        text_count, skipped_text_count = self._import_text_batches(
            collection_name=text_collection,
            documents=text_docs,
            batch_size=text_batch_size,
            max_workers=text_embed_workers,
            tracking_file=tracking_file,
            progress_callback=progress_callback,
        )
        image_count, skipped_image_count = self._import_image_batches(
            collection_name=image_collection,
            documents=image_docs,
            image_paths=image_paths,
            batch_size=image_batch_size,
            max_workers=image_embed_workers,
            tracking_file=tracking_file,
            progress_callback=progress_callback,
        )

        return {
            "pdf_path": pdf_path,
            "export_dir": export_dir,
            "text_count": text_count,
            "image_count": image_count,
            "skipped_text_count": skipped_text_count,
            "skipped_image_count": skipped_image_count,
            "tracking_file": tracking_file,
            "summary": structured_data.get("summary", {}),
        }

    def index_pattern(
        self,
        pattern: str,
        output_root: str,
        text_collection: str,
        image_collection: str,
        text_batch_size: int = 8,
        text_embed_workers: int = 4,
        image_batch_size: int = 1,
        image_embed_workers: int = 2,
        tracking_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        pdf_paths = sorted(glob.glob(pattern))
        results = []
        for pdf_path in pdf_paths:
            results.append(
                self.index_pdf(
                    pdf_path=pdf_path,
                    output_root=output_root,
                    text_collection=text_collection,
                    image_collection=image_collection,
                    text_batch_size=text_batch_size,
                    text_embed_workers=text_embed_workers,
                    image_batch_size=image_batch_size,
                    image_embed_workers=image_embed_workers,
                    tracking_file=tracking_file,
                )
            )
        return results
