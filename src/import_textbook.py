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


def embed_batch(texts: List[str], model: str) -> List[List[float]]:
    """调用 DashScope 生成一批文本的嵌入向量"""
    resp = dashscope.TextEmbedding.call(
        model=model,
        input=texts,
        text_type="document",
    )
    if resp.status_code != 200:
        raise RuntimeError(f"嵌入 API 错误: {resp.code} - {resp.message}")
    return [item["embedding"] for item in resp.output["embeddings"]]


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


def main():
    # 配置
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        logger.error("请在 .env 中设置 DASHSCOPE_API_KEY")
        sys.exit(1)
    dashscope.api_key = api_key

    embed_model = os.getenv("EMBED_MODEL", "text-embedding-v4")
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name = os.getenv("TUTOR_COLLECTION", "textbook_embeddings")
    textbook_dir = os.getenv("TUTOR_TEXTBOOK_DIR", "./textbook")
    tracking_file = os.path.join(persist_dir, f"{collection_name}_tracking.json")

    chunk_size = int(os.getenv("TUTOR_CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("TUTOR_CHUNK_OVERLAP", "200"))

    # 检查目录
    if not os.path.isdir(textbook_dir):
        logger.error("课本目录不存在: %s，请创建并放入 PDF 文件", textbook_dir)
        sys.exit(1)

    pdf_files = [f for f in os.listdir(textbook_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.error("课本目录 %s 下没有 PDF 文件", textbook_dir)
        sys.exit(1)

    logger.info("找到 %d 个 PDF 文件: %s", len(pdf_files), pdf_files)

    # 选择导入模式
    print("\n请选择导入模式:")
    print("  1. 全量导入（清空已有数据，重新导入）")
    print("  2. 增量更新（只导入新增内容）")
    choice = input("\n请输入 (1/2) [默认 2]: ").strip() or "2"

    # 加载 PDF 并分块
    from src.utils.doc_loader import PDFLoader
    loader = PDFLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = loader.load_directory(textbook_dir)

    if not all_chunks:
        logger.warning("没有提取到任何文本块")
        sys.exit(0)

    logger.info("共提取 %d 个文本块", len(all_chunks))

    # 初始化 ChromaDB
    client = chromadb.PersistentClient(path=persist_dir)

    if choice == "1":
        # 全量导入：删除已有集合
        try:
            client.delete_collection(collection_name)
            logger.info("已删除旧集合: %s", collection_name)
        except Exception:
            pass
        existing_hashes = set()
    else:
        existing_hashes = load_tracking(tracking_file)
        logger.info("已有 %d 条记录", len(existing_hashes))

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # 过滤已导入的块
    new_chunks = []
    for chunk in all_chunks:
        h = get_text_hash(chunk.text)
        if h not in existing_hashes:
            new_chunks.append((chunk, h))

    if not new_chunks:
        logger.info("没有新内容需要导入")
        return

    logger.info("需要导入 %d 个新文本块（跳过 %d 个已有）",
                len(new_chunks), len(all_chunks) - len(new_chunks))

    # 分批生成嵌入并写入
    total = len(new_chunks)
    imported = 0
    failed = 0
    all_hashes = set(existing_hashes)

    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = new_chunks[i:i + EMBED_BATCH_SIZE]
        texts = [c.text for c, _ in batch]

        try:
            embeddings = embed_batch(texts, embed_model)
        except Exception as e:
            logger.error("批次 %d-%d 嵌入失败: %s", i, i + len(batch), e)
            failed += len(batch)
            time.sleep(1)
            continue

        # 写入 ChromaDB
        ids = []
        documents = []
        metadatas = []
        for j, (chunk, h) in enumerate(batch):
            ids.append(f"textbook_{i + j}_{h[:8]}")
            documents.append(chunk.text)
            metadatas.append(chunk.metadata)
            all_hashes.add(h)

        try:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            imported += len(batch)
        except Exception as e:
            logger.error("写入 ChromaDB 失败: %s", e)
            failed += len(batch)

        # 进度
        progress = (i + len(batch)) / total * 100
        logger.info("进度: %.1f%% (%d/%d)", progress, i + len(batch), total)

        # 避免 API 限流
        time.sleep(0.5)

    # 保存跟踪数据
    save_tracking(tracking_file, all_hashes)

    logger.info("导入完成！成功 %d，失败 %d，总计 %d", imported, failed, total)
    logger.info("集合 '%s' 当前共 %d 条记录", collection_name, collection.count())


if __name__ == "__main__":
    main()
