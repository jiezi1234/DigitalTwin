"""
课本导入脚本
读取 textbook/ 目录下的 PDF 文件，分块后生成向量嵌入，存入 ChromaDB
"""

import os
import sys
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from dotenv import load_dotenv
import dashscope
import chromadb
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# DashScope 嵌入 API 每次最多 10 条
EMBED_BATCH_SIZE = 10
MAX_WORKERS = 4


def get_text_hash(text: str) -> str:
    """生成文本的 MD5 哈希，用于去重"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def embed_batch_with_retry(texts: List[str], model: str, max_retries: int = 3) -> List[List[float]]:
    """调用 DashScope 生成嵌入，含指数退避重试"""
    for attempt in range(max_retries):
        try:
            resp = dashscope.TextEmbedding.call(
                model=model,
                input=texts,
                text_type="document",
            )
            if resp.status_code != 200:
                raise RuntimeError(f"嵌入 API 错误: {resp.code} - {resp.message}")
            return [item["embedding"] for item in resp.output["embeddings"]]
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            err_str = str(e)
            wait = 4 if ("429" in err_str or "rate" in err_str.lower()) else 2
            logger.warning("嵌入重试 %d/%d: %s，等待 %ds", attempt + 1, max_retries, e, wait)
            time.sleep(wait)
    return []  # unreachable


def load_tracking(tracking_file: str) -> set:
    """加载已导入的文本哈希集合"""
    if os.path.exists(tracking_file):
        with open(tracking_file, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_tracking(tracking_file: str, hashes: set):
    """保存已导入的文本哈希集合"""
    with open(tracking_file, "w", encoding="utf-8") as f:
        json.dump(list(hashes), f)



def process_chunks(chunks: List[Any], existing_hashes: set, embed_model: str, collection: Any, max_workers: int):
    """通用处理流程：过滤 -> 嵌入 -> 写入"""
    new_chunks = []
    for chunk in chunks:
        h = get_text_hash(chunk.text)
        if h not in existing_hashes:
            new_chunks.append((chunk, h))

    if not new_chunks:
        logger.info("没有新内容需要导入")
        return 0, 0, len(chunks)

    total = len(new_chunks)
    logger.info("需要导入 %d 个新文本块（跳过 %d 个已有）", total, len(chunks) - total)

    imported = 0
    failed = 0
    
    # ------- 阶段一：并行生成所有嵌入向量 -------
    batches = [new_chunks[i:i + EMBED_BATCH_SIZE] for i in range(0, total, EMBED_BATCH_SIZE)]
    all_results: list = [None] * len(batches)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(embed_batch_with_retry, [c.text for c, _ in batch], embed_model): i
            for i, batch in enumerate(batches)
        }
        done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            done += 1
            try:
                vecs = future.result()
                all_results[idx] = (batches[idx], vecs)
            except Exception as e:
                logger.error("批次 %d 嵌入失败: %s", idx, e)
                failed += len(batches[idx])
            if done % 10 == 0 or done == len(batches):
                logger.info("嵌入进度: %d/%d 批次", done, len(batches))

    # ------- 阶段二：顺序 upsert 写入 -------
    for i, result in enumerate(all_results):
        if result is None: continue
        batch, vecs = result
        ids, docs, metas, embeddings = [], [], [], []
        for j, ((chunk, h), vec) in enumerate(zip(batch, vecs)):
            ids.append(f"textbook_{int(time.time())}_{i*EMBED_BATCH_SIZE+j}_{h[:8]}")
            docs.append(chunk.text)
            metas.append(chunk.metadata)
            embeddings.append(vec)
            existing_hashes.add(h)

        try:
            collection.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
            imported += len(batch)
        except Exception as e:
            logger.error("写入失败（批次 %d）: %s", i, e)
            failed += len(batch)
            
    return imported, failed, total


def main():
    # 基础配置
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        logger.error("请在 .env 中设置 DASHSCOPE_API_KEY"); sys.exit(1)
    dashscope.api_key = api_key

    embed_model = os.getenv("EMBED_MODEL", "text-embedding-v4")
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    textbook_dir = os.getenv("TUTOR_TEXTBOOK_DIR", "./textbook")

    chunk_size = int(os.getenv("TUTOR_CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("TUTOR_CHUNK_OVERLAP", "200"))

    if not os.path.isdir(textbook_dir):
        logger.error("目录不存在: %s", textbook_dir); sys.exit(1)

    client = chromadb.PersistentClient(path=persist_dir)

    # 1. 集合选择
    print("\n--- 向量数据库集合选择 ---")
    try:
        collections = client.list_collections()
        if collections:
            print("现有集合：")
            for i, col in enumerate(collections, 1):
                print(f"  {i}. {col.name} ({col.count()} 条记录)")
            print(f"  {len(collections) + 1}. 创建新集合")
            choice = input(f"请选择 (1-{len(collections) + 1}) [默认 1]: ").strip() or "1"
            if choice.isdigit() and 1 <= int(choice) <= len(collections):
                collection_name = collections[int(choice) - 1].name
            else:
                collection_name = input("请输入新集合名称: ").strip() or "textbook_embeddings"
        else:
            collection_name = input("请输入新集合名称 [默认 textbook_embeddings]: ").strip() or "textbook_embeddings"
    except Exception:
        collection_name = "textbook_embeddings"

    # 2. 文件选择
    from pathlib import Path
    textbook_path = Path(textbook_dir)
    pdf_files = sorted(list(textbook_path.glob("*.pdf")))
    if not pdf_files:
        logger.error("目录下无 PDF"); sys.exit(1)

    print("\n--- PDF 文件选择 ---")
    for f in pdf_files: print(f"  {f.name}")
    while True:
        pattern = input("\n文件名匹配模式 (如 *.pdf) [默认 *.pdf]: ").strip() or "*.pdf"
        matched_files = sorted(list(textbook_path.glob(pattern)))
        if matched_files: break
        print("未匹配到文件，请重试")

    # 3. 导入模式
    print("\n--- 导入模式 ---")
    print("  1. 全量导入 (清空集合重新开始)")
    print("  2. 增量更新 (只补全缺失 chunk)")
    is_full = (input("\n选择 (1/2) [默认 2]: ").strip() == "1")

    if is_full:
        try: client.delete_collection(collection_name)
        except Exception: pass
        existing_hashes = set()
    else:
        tracking_file = os.path.join(persist_dir, f"{collection_name}_tracking.json")
        existing_hashes = load_tracking(tracking_file)

    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    print("\n请设置并发嵌入线程数：")
    w = input("线程数 [默认 4]: ").strip() or "4"
    max_workers = int(w) if w.isdigit() else 4

    # 4. 逐个文件处理 (避免内存爆炸)
    from src.utils.doc_loader import PDFLoader
    ocr_workers = int(os.getenv("TUTOR_OCR_WORKERS", "0"))
    loader = PDFLoader(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        ocr_enabled=True, ocr_language=os.getenv("TUTOR_OCR_LANGUAGE", "chi_sim+eng"),
        ocr_dpi=int(os.getenv("TUTOR_OCR_DPI", "150")),
        ocr_text_threshold=int(os.getenv("TUTOR_OCR_TEXT_THRESHOLD", "50")),
        ocr_workers=ocr_workers
    )

    total_imported = 0
    total_failed = 0

    for i, f_path in enumerate(matched_files):
        print(f"\n[{i+1}/{len(matched_files)}] 正在处理: {f_path.name}")
        try:
            # 加载一个 PDF 的所有 chunk (约几 MB)
            chunks = loader.load_pdf(str(f_path))
            # 立即执行 嵌入+写入
            imp, fail, _ = process_chunks(chunks, existing_hashes, embed_model, collection, max_workers)
            total_imported += imp
            total_failed += fail
            # 每个文件处理完保存一次追踪，防止意外中断
            tracking_file = os.path.join(persist_dir, f"{collection_name}_tracking.json")
            save_tracking(tracking_file, existing_hashes)
        except Exception as e:
            logger.error("文件 %s 处理失败: %s", f_path.name, e)

    print(f"\n--- 导入总结 ---")
    print(f"文件数: {len(matched_files)}")
    print(f"成功导入: {total_imported} chunk")
    print(f"失败: {total_failed} chunk")
    print(f"集合 '{collection_name}' 当前总记录: {collection.count()}")


if __name__ == "__main__":
    main()
