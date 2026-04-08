"""
课程 PDF 导入脚本

notes1_2022.pdf / notes7_2022.pdf:
  走多模态文本块 + 图片向量导入

textbook.pdf:
  走 OCR 文本提取后再文本向量化导入
"""

import argparse
import concurrent.futures
import json
import os
import threading
from typing import Dict, List

from src.api.config import Config
from src.infrastructure.db_client import DBClient
from src.infrastructure.document import Document
from src.infrastructure.multimodal_embedding_client import MultiModalEmbeddingClient
from src.infrastructure.text_embedding_client import TextEmbeddingClient
from src.loaders.pdf_loader import PDFLoader
from src.services.multimodal_pdf_service import MultiModalPDFIndexService

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导入课程 PDF 到向量库")
    default_notes_files = [
        os.path.join(Config.PROJECT_ROOT, "data/pdf/notes1_2022.pdf"),
        os.path.join(Config.PROJECT_ROOT, "data/pdf/notes7_2022.pdf"),
    ]
    default_textbook_file = os.path.join(Config.PROJECT_ROOT, "data/pdf/textbook.pdf")

    parser.add_argument(
        "--persist-dir",
        default=Config.CHROMA_PERSIST_DIR,
        help="Chroma 持久化目录",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(Config.PROJECT_ROOT, "output", "course_mm"),
        help="多模态结构化导出根目录",
    )
    parser.add_argument(
        "--notes-files",
        nargs="*",
        default=None,
        help="走多模态导入的 PDF 文件列表",
    )
    parser.add_argument(
        "--textbook-file",
        default=default_textbook_file,
        help="走 OCR 文本导入的教材 PDF",
    )
    parser.add_argument(
        "--skip-textbook",
        action="store_true",
        help="跳过 textbook OCR 导入",
    )
    parser.add_argument(
        "--mm-text-collection",
        default=Config.TUTOR_MM_TEXT_COLLECTION,
        help="多模态文本 collection",
    )
    parser.add_argument(
        "--mm-image-collection",
        default=Config.TUTOR_MM_IMAGE_COLLECTION,
        help="多模态图片 collection",
    )
    parser.add_argument(
        "--ocr-text-collection",
        default=Config.TUTOR_OCR_TEXT_COLLECTION,
        help="OCR 文本 collection",
    )
    parser.add_argument(
        "--mm-model",
        default=Config.MM_EMBED_MODEL,
        help="多模态 embedding 模型",
    )
    parser.add_argument(
        "--text-embed-model",
        default=Config.EMBED_MODEL,
        help="OCR 文本 embedding 模型",
    )
    parser.add_argument(
        "--ocr-language",
        default="chi_sim+eng",
        help="OCR 语言配置",
    )
    parser.add_argument(
        "--tracking-dir",
        default=None,
        help="增量导入 tracking 目录，默认在 persist-dir 下自动创建",
    )
    parser.add_argument(
        "--pdf-workers",
        type=int,
        default=2,
        help="多模态 PDF 并行导入数",
    )
    parser.add_argument(
        "--mm-text-batch-size",
        type=int,
        default=8,
        help="多模态文本 embedding 批大小",
    )
    parser.add_argument(
        "--mm-text-workers",
        type=int,
        default=4,
        help="多模态文本 embedding 并发数",
    )
    parser.add_argument(
        "--mm-image-batch-size",
        type=int,
        default=1,
        help="多模态图片 embedding 批大小",
    )
    parser.add_argument(
        "--mm-image-workers",
        type=int,
        default=2,
        help="多模态图片 embedding 并发数",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="导入前清空目标 collections",
    )
    parser.set_defaults(
        _default_notes_files=default_notes_files,
        _default_textbook_file=default_textbook_file,
    )
    return parser


def _validate_files(paths: List[str]) -> None:
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"以下文件不存在: {missing}")


def _load_json_file(path: str, default: Dict) -> Dict:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json_file(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _index_single_multimodal_pdf(args, pdf_path: str, tracking_dir: str, progress_callback=None) -> dict:
    db_client = DBClient(
        persist_dir=args.persist_dir,
        embed_model=args.text_embed_model,
    )
    service = MultiModalPDFIndexService(
        db_client=db_client,
        embedding_client=MultiModalEmbeddingClient(model=args.mm_model),
    )
    tracking_file = os.path.join(
        tracking_dir,
        f"{os.path.splitext(os.path.basename(pdf_path))[0]}_multimodal_tracking.json",
    )
    return service.index_pdf(
        pdf_path=pdf_path,
        output_root=args.output_root,
        text_collection=args.mm_text_collection,
        image_collection=args.mm_image_collection,
        text_batch_size=args.mm_text_batch_size,
        text_embed_workers=args.mm_text_workers,
        image_batch_size=args.mm_image_batch_size,
        image_embed_workers=args.mm_image_workers,
        tracking_file=tracking_file,
        progress_callback=progress_callback,
    )


def _build_documents_from_pages(loader: PDFLoader, pages_data: List[Dict], filename: str) -> List[Document]:
    documents: List[Document] = []
    current_chapter = "前言"
    current_section = ""

    for p_data in pages_data:
        if not p_data or p_data.get("error") or not p_data.get("text"):
            continue

        text = p_data["text"]
        page_idx = p_data["page_num"]
        is_ocr = p_data["is_ocr"]

        current_chapter, current_section = loader._detect_headings(text, current_chapter, current_section)
        page_chunks = loader._split_text(text)
        for i, chunk_text in enumerate(page_chunks):
            metadata = {
                "source": "pdf",
                "source_file": filename,
                "page": page_idx + 1,
                "ocr": is_ocr,
                "chapter": current_chapter,
                "section": current_section,
                "chunk_index": i,
                "content_type": "textbook",
            }
            documents.append(Document(content=chunk_text, metadata=metadata))

    return documents


def _ocr_textbook_with_progress(
    pdf_path: str,
    tracking_file: str,
    ocr_language: str,
    progress_callback=None,
) -> List[Dict]:
    loader = PDFLoader(
        filepath=pdf_path,
        ocr_enabled=True,
        ocr_language=ocr_language,
    )
    tracking = _load_json_file(
        tracking_file,
        {"completed_pages": [], "page_results": {}, "completed_doc_ids": []},
    )

    import fitz

    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        if loader.ocr_enabled and total_pages > 0:
            try:
                doc[0].get_textpage_ocr(language=loader.ocr_language, dpi=72, full=False)
            except Exception:
                pass

    completed_pages = set(tracking.get("completed_pages", []))
    page_results = tracking.get("page_results", {})
    pending_pages = [page_num for page_num in range(total_pages) if page_num not in completed_pages]

    if progress_callback:
        progress_callback(event="ocr_start", total_pages=total_pages, completed_pages=len(completed_pages))

    tasks = [
        {
            "filepath": pdf_path,
            "page_num": page_num,
            "ocr_enabled": True,
            "ocr_language": loader.ocr_language,
            "ocr_text_threshold": loader.ocr_text_threshold,
        }
        for page_num in pending_pages
    ]

    if tasks:
        max_workers = loader.max_workers
        if max_workers <= 1 or len(tasks) <= 1:
            results = [PDFLoader._process_single_page_static(task) for task in tasks]
            for result in results:
                page_num = result["page_num"]
                completed_pages.add(page_num)
                page_results[str(page_num)] = result
                tracking["completed_pages"] = sorted(completed_pages)
                tracking["page_results"] = page_results
                _save_json_file(tracking_file, tracking)
                if progress_callback:
                    progress_callback(event="ocr_page_done", page_num=page_num + 1)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_page = {
                    executor.submit(PDFLoader._process_single_page_static, task): task["page_num"]
                    for task in tasks
                }
                for future in concurrent.futures.as_completed(future_to_page):
                    result = future.result()
                    page_num = result["page_num"]
                    completed_pages.add(page_num)
                    page_results[str(page_num)] = result
                    tracking["completed_pages"] = sorted(completed_pages)
                    tracking["page_results"] = page_results
                    _save_json_file(tracking_file, tracking)
                    if progress_callback:
                        progress_callback(event="ocr_page_done", page_num=page_num + 1)

    ordered_pages = []
    for page_num in range(total_pages):
        page_result = page_results.get(str(page_num))
        if page_result is not None:
            ordered_pages.append(page_result)
    return ordered_pages


def _import_ocr_documents_incremental(
    db_client: DBClient,
    documents: List[Document],
    collection_name: str,
    tracking_file: str,
    batch_size: int = 10,
    max_workers: int = 4,
    text_embedding_client: TextEmbeddingClient = None,
    progress_callback=None,
) -> Dict:
    tracking = _load_json_file(
        tracking_file,
        {"completed_pages": [], "page_results": {}, "completed_doc_ids": []},
    )
    completed_doc_ids = set(tracking.get("completed_doc_ids", []))
    pending_documents = [
        doc for doc in documents
        if (doc.doc_id or PDFLoader.generate_hash(doc.content, doc.metadata)) not in completed_doc_ids
    ]

    if progress_callback:
        progress_callback(
            event="embed_start",
            total_docs=len(documents),
            pending_docs=len(pending_documents),
        )

    if not pending_documents:
        return {"count": 0, "status": "skipped", "duration": 0.0}

    text_embedding_client = text_embedding_client or TextEmbeddingClient(model=Config.EMBED_MODEL)
    chunks = [pending_documents[i:i + batch_size] for i in range(0, len(pending_documents), batch_size)]
    all_embeddings = [None] * len(chunks)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(text_embedding_client.embed_texts, [d.content for d in chunk]): idx
            for idx, chunk in enumerate(chunks)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            all_embeddings[idx] = future.result()

    imported_count = 0
    for idx, embeddings in enumerate(all_embeddings):
        chunk = chunks[idx]
        chunk_ids = [doc.doc_id or PDFLoader.generate_hash(doc.content, doc.metadata) for doc in chunk]
        chunk_count = db_client.bulk_import(
            collection_name=collection_name,
            documents=[doc.content for doc in chunk],
            embeddings=embeddings,
            metadatas=[doc.metadata for doc in chunk],
            ids=chunk_ids,
        )
        imported_count += chunk_count
        completed_doc_ids.update(chunk_ids)
        tracking["completed_doc_ids"] = sorted(completed_doc_ids)
        _save_json_file(tracking_file, tracking)
        if progress_callback:
            progress_callback(
                event="embed_batch_done",
                batch_size=len(chunk),
                imported_count=imported_count,
            )

    return {"count": imported_count, "status": "success", "duration": 0.0}


def main() -> None:
    args = build_parser().parse_args()

    notes_files = args.notes_files if args.notes_files is not None else args._default_notes_files
    textbook_file = None if args.skip_textbook else args.textbook_file

    selected_files = list(notes_files)
    if textbook_file:
        selected_files.append(textbook_file)
    if not selected_files:
        raise ValueError("未选择任何导入文件。请提供 --notes-files 或 --textbook-file，或取消 --skip-textbook。")

    _validate_files(selected_files)

    os.makedirs(args.persist_dir, exist_ok=True)
    os.makedirs(args.output_root, exist_ok=True)
    tracking_dir = args.tracking_dir or os.path.join(args.persist_dir, "import_tracking")
    os.makedirs(tracking_dir, exist_ok=True)

    mm_db_client = DBClient(
        persist_dir=args.persist_dir,
        embed_model=args.text_embed_model,
    )

    ocr_db_client = DBClient(
        persist_dir=args.persist_dir,
        embed_model=args.text_embed_model,
    )

    if args.reset:
        mm_db_client.delete_collection(args.mm_text_collection)
        mm_db_client.delete_collection(args.mm_image_collection)
        ocr_db_client.delete_collection(args.ocr_text_collection)
        for tracking_name in os.listdir(tracking_dir):
            if tracking_name.endswith(".json"):
                os.remove(os.path.join(tracking_dir, tracking_name))

    mm_results = []
    total_steps = len(notes_files) + (1 if textbook_file else 0)
    progress = tqdm(total=total_steps, desc="课程资料导入", unit="pdf", position=0) if tqdm else None
    progress_lock = threading.Lock()
    pdf_bars = {}

    def make_progress_callback(pdf_path: str, position: int):
        pdf_name = os.path.basename(pdf_path)

        def callback(**event):
            if not tqdm:
                return
            with progress_lock:
                bar = pdf_bars.get(pdf_path)
                if event.get("stage") == "prepare" and event.get("event") == "ready":
                    total_units = event.get("text_total", 0) + event.get("image_total", 0)
                    if total_units <= 0:
                        total_units = 1
                    bar = tqdm(
                        total=total_units,
                        desc=pdf_name,
                        unit="item",
                        position=position,
                        leave=True,
                    )
                    pdf_bars[pdf_path] = bar
                    bar.set_postfix_str(
                        f"text={event.get('text_total', 0)} image={event.get('image_total', 0)}"
                    )
                    return

                if bar is None:
                    return

                if event.get("event") == "batch_done":
                    bar.update(event.get("batch_size", 0))
                    bar.set_postfix_str(
                        f"{event.get('stage')} imported={event.get('imported_count', 0)}"
                    )
                elif event.get("event") == "skipped_all":
                    bar.set_postfix_str(f"{event.get('stage')} skipped")

        return callback

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.pdf_workers)) as executor:
        future_to_pdf = {
            executor.submit(
                _index_single_multimodal_pdf,
                args,
                pdf_path,
                tracking_dir,
                make_progress_callback(pdf_path, idx + 1),
            ): pdf_path
            for idx, pdf_path in enumerate(notes_files)
        }
        for future in concurrent.futures.as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            if progress:
                progress.set_postfix_str(f"多模态完成: {os.path.basename(pdf_path)}")
            mm_results.append(future.result())
            if progress:
                progress.update(1)
            if tqdm:
                with progress_lock:
                    bar = pdf_bars.get(pdf_path)
                    if bar is not None:
                        bar.close()

    mm_results.sort(key=lambda item: item["pdf_path"])

    ocr_result = {"count": 0, "status": "skipped", "duration": 0.0}
    ocr_bar = None
    ocr_tracking_file = os.path.join(tracking_dir, "textbook_ocr_tracking.json")

    def ocr_progress_callback(**event):
        nonlocal ocr_bar
        if not tqdm:
            return
        with progress_lock:
            if event.get("event") == "ocr_start":
                total_pages = event.get("total_pages", 0)
                completed_pages = event.get("completed_pages", 0)
                ocr_bar = tqdm(
                    total=total_pages,
                    initial=completed_pages,
                    desc=os.path.basename(textbook_file),
                    unit="page",
                    position=len(notes_files) + 1,
                    leave=True,
                )
                ocr_bar.set_postfix_str("OCR")
            elif event.get("event") == "ocr_page_done" and ocr_bar is not None:
                ocr_bar.update(1)
                ocr_bar.set_postfix_str(f"OCR page={event.get('page_num')}")
            elif event.get("event") == "embed_start" and ocr_bar is not None:
                ocr_bar.set_postfix_str(
                    f"Embed pending={event.get('pending_docs', 0)}/{event.get('total_docs', 0)}"
                )
            elif event.get("event") == "embed_batch_done" and ocr_bar is not None:
                ocr_bar.set_postfix_str(f"Embed imported={event.get('imported_count', 0)}")

    if textbook_file:
        if progress:
            progress.set_postfix_str(f"OCR: {os.path.basename(textbook_file)}")
        ocr_loader = PDFLoader(
            filepath=textbook_file,
            ocr_enabled=True,
            ocr_language=args.ocr_language,
        )
        pages_data = _ocr_textbook_with_progress(
            pdf_path=textbook_file,
            tracking_file=ocr_tracking_file,
            ocr_language=args.ocr_language,
            progress_callback=ocr_progress_callback,
        )
        ocr_documents = _build_documents_from_pages(
            loader=ocr_loader,
            pages_data=pages_data,
            filename=os.path.basename(textbook_file),
        )
        ocr_result = _import_ocr_documents_incremental(
            db_client=ocr_db_client,
            documents=ocr_documents,
            collection_name=args.ocr_text_collection,
            tracking_file=ocr_tracking_file,
            batch_size=10,
            max_workers=4,
            text_embedding_client=TextEmbeddingClient(model=args.text_embed_model),
            progress_callback=ocr_progress_callback,
        )
        if progress:
            progress.update(1)
        if ocr_bar is not None:
            ocr_bar.close()

    if progress:
        progress.close()

    print(json.dumps({
        "persist_dir": os.path.abspath(args.persist_dir),
        "output_root": os.path.abspath(args.output_root),
        "tracking_dir": os.path.abspath(tracking_dir),
        "multimodal": {
            "text_collection": args.mm_text_collection,
            "image_collection": args.mm_image_collection,
            "files": mm_results,
        },
        "ocr_text": {
            "collection": args.ocr_text_collection,
            "file": textbook_file,
            "result": {
                "count": ocr_result.get("count", 0),
                "status": ocr_result.get("status"),
                "duration": ocr_result.get("duration"),
            },
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
