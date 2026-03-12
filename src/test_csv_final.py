# 数据导入脚本，使用 ChromaDB 本地向量数据库（无需安装任何数据库服务）
import os
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import dashscope
dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings.dashscope import DashScopeEmbeddings

from .core.persona_manager import PersonaManager
from .utils.csv_loader import WeChatCSVLoader
from .utils.tracking import load_import_tracking, save_import_tracking

# ChromaDB 本地持久化目录
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# 增量跟踪文件（与向量库同目录，避免不同工作目录时丢失）
IMPORT_TRACKING_FILE = os.path.join(CHROMA_PERSIST_DIR, "import_tracking.json")

# PersonaManager 实例
pm = PersonaManager(CHROMA_PERSIST_DIR)

llm = ChatTongyi(model=os.getenv("CHAT_MODEL", "qwen-plus"))


def compute_max_tokens(docs, percentile=80, scale=1.5, minimum=30):
    """采样 self 消息长度，估算合适的 max_tokens。

    中文字符在 DashScope tokenizer 中约 1-1.5 token/字，乘以 scale 留余量。
    取 P{percentile} 而非均值，避免被少量超长消息拉高。
    """
    self_lengths = [
        len(d.metadata.get("msg_content", ""))
        for d in docs
        if str(d.metadata.get("is_sender", "0")) == "1"
    ]
    if not self_lengths:
        return 150
    self_lengths.sort()
    idx = min(int(len(self_lengths) * percentile / 100), len(self_lengths) - 1)
    p_val = self_lengths[idx]
    result = max(minimum, int(p_val * scale))
    print(f"  self 消息共 {len(self_lengths)} 条，P{percentile} 长度 = {p_val} 字，"
          f"max_tokens = {result}")
    return result


def _embed_chunk(embeddings_model, texts, chunk_idx, max_retries=3):
    """线程安全：嵌入单个文本块，含重试"""
    for attempt in range(max_retries):
        try:
            return chunk_idx, embeddings_model.embed_documents(texts)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate" in err_str.lower() or "throttl" in err_str.lower()
            time.sleep((2 ** (attempt + 1)) if is_rate_limit else 1)


def create_vectorstore_with_progress(
    documents, embeddings, collection_name,
    embed_chunk_size=10,
    chroma_write_batch=500,
    incremental=False,
    max_workers=4,
):
    """并行生成嵌入，再顺序写入 ChromaDB（绕过 LangChain 二次嵌入）

    Args:
        documents: 要导入的文档列表
        embeddings: embedding 模型
        collection_name: ChromaDB 集合名称
        embed_chunk_size: 每次调用 embed_documents 的文本数（DashScope v4 每次 API 请求上限 10 条）
        chroma_write_batch: ChromaDB 批量写入大小
        incremental: 是否为增量更新模式（True=追加，False=清空重建）
        max_workers: 并发线程数
    """
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, desc="Processing", total=None):
            print(f"{desc}...")
            return iterable

    print(f"开始{'增量更新' if incremental else '全量导入'}向量数据库，共 {len(documents)} 个文档...")
    print(f"ChromaDB 持久化目录: {CHROMA_PERSIST_DIR}")
    print(f"集合名称: {collection_name}，并发线程: {max_workers}，每块大小: {embed_chunk_size}")

    try:
        # ── 阶段一：并行生成所有嵌入 ─────────────────────────────────────
        chunks = [documents[i:i + embed_chunk_size] for i in range(0, len(documents), embed_chunk_size)]
        all_results = [None] * len(chunks)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_embed_chunk, embeddings, [d.page_content for d in chunk], i): i
                for i, chunk in enumerate(chunks)
            }
            for future in tqdm(as_completed(futures), total=len(chunks), desc="生成嵌入向量"):
                try:
                    idx, vecs = future.result()
                    all_results[idx] = (chunks[idx], vecs)
                except Exception as e:
                    print(f"⚠️ 块 {futures[future]} 嵌入失败，跳过: {e}")

        # ── 阶段二：顺序写入 ChromaDB ────────────────────────────────────
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        if not incremental:
            try:
                client.delete_collection(collection_name)
                print(f"✅ 已清空旧集合 '{collection_name}'")
            except Exception:
                pass
        collection = client.get_or_create_collection(collection_name)

        # 拍平所有结果，生成稳定 ID（md5 去重，源数据中完全相同的记录只保留一条）
        flat_docs, flat_vecs, flat_metas, flat_ids = [], [], [], []
        seen_ids = set()
        dup_count = 0
        for item in all_results:
            if item is None:
                continue
            chunk_docs, vecs = item
            for doc, vec in zip(chunk_docs, vecs):
                uid = hashlib.md5(
                    f"{doc.metadata.get('source', '')}"
                    f"{doc.metadata.get('chat_time', '')}"
                    f"{doc.metadata.get('msg_content', '')}".encode()
                ).hexdigest()
                if uid in seen_ids:
                    dup_count += 1
                    continue
                seen_ids.add(uid)
                flat_docs.append(doc.page_content)
                flat_vecs.append(vec)
                flat_metas.append(doc.metadata)
                flat_ids.append(uid)
        if dup_count:
            print(f"⚠️ 跳过 {dup_count} 条重复记录（源数据中内容完全相同）")

        for i in tqdm(range(0, len(flat_docs), chroma_write_batch), desc="写入向量数据库"):
            end = i + chroma_write_batch
            collection.upsert(
                documents=flat_docs[i:end],
                embeddings=flat_vecs[i:end],
                metadatas=flat_metas[i:end],
                ids=flat_ids[i:end],
            )

        print(f"✅ 成功写入 {len(flat_docs)} 个文档到集合 '{collection_name}'")
        return True

    except Exception as e:
        print(f"❌ 向量数据库创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def interactive_chat(rag_chain):
    """交互式聊天函数"""
    print("\n" + "="*60)
    print("WeChat Chat RAG System Ready!")
    print("="*60)
    print("Tip: You can ask these types of questions:")
    print("   - 某个人说了什么？")
    print("   - 关于某个话题的聊天内容")
    print("   - 某个时间段的对话")
    print("   - 聊天记录的统计信息")
    print("="*60)

    test_queries = ["聊天记录中都有哪些人参与了对话？", "最近在聊什么话题？"]
    for test_query in test_queries:
        print(f"\nTest query: {test_query}")
        try:
            result = rag_chain.invoke(test_query)
            print(f"Result: {result}")
            break
        except Exception as e:
            print(f"Error: Test query failed: {e}")
            continue

    while True:
        try:
            query = input("\n❓ 请输入您的问题（输入'quit'、'exit'或'q'退出）: ").strip()
            if query.lower() in ['quit', 'exit', 'q', '退出']:
                print("👋 再见！")
                break
            if not query:
                print("Warning: Please enter a valid question")
                continue

            print(f"\nQuerying: {query}")
            print("-" * 40)
            start_time = time.time()
            result = rag_chain.invoke(query)
            print(f"Answer: {result}")
            print(f"Time: {time.time() - start_time:.2f} seconds")

        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见！")
            break
        except Exception as e:
            print(f"Error: Query failed: {e}")
            print("Tip: Please try rephrasing your question")


def main():
    """主程序"""
    try:
        print("Starting WeChat Chat RAG System...")

        csv_dir_path = "csv_clean" if os.path.exists("csv_clean") else "csv"
        if not os.path.exists(csv_dir_path):
            print("Error: csv_clean/ 和 csv/ 目录均不存在，请先运行 preprocess_csv.py")
            return

        # ── 分身选择 ──────────────────────────────────────────────────────
        existing_personas = pm.list()
        persona = None

        if existing_personas:
            print("\n现有分身：")
            for i, p in enumerate(existing_personas, 1):
                print(f"  {i}. {p['name']}（集合: {p['collection']}，已导入: {p['doc_count']} 条）")
            print(f"  {len(existing_personas) + 1}. 创建新分身")

            while True:
                choice = input(f"请选择 (1-{len(existing_personas) + 1}): ").strip()
                if choice.isdigit():
                    idx = int(choice)
                    if 1 <= idx <= len(existing_personas):
                        persona = existing_personas[idx - 1]
                        print(f"已选择分身：{persona['name']}")
                        break
                    elif idx == len(existing_personas) + 1:
                        persona = None
                        break
                print("无效选项，请重新输入")

        if persona is None:
            print("\n创建新分身：")
            while True:
                name = input("请输入分身名称: ").strip()
                if name:
                    break
                print("名称不能为空")
            print("请输入系统提示词（描述这个数字分身的角色和语气，输入空行结束）：")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            system_prompt = "\n".join(lines) if lines else (
                "你需要扮演聊天记录中标签'发送者'的值为'self'的那个人，"
                "你是ta的数字克隆人，模仿ta的语气用词，以第一人称交流。"
                "用自然、口语化、简洁的中文回答。"
            )
            persona = pm.create(name=name, system_prompt=system_prompt)
            print(f"✅ 已创建分身：{persona['name']}（集合: {persona['collection']}）")

        CHROMA_COLLECTION = persona["collection"]
        persona_id = persona["id"]
        # ─────────────────────────────────────────────────────────────────

        # ── CSV 文件选择 ───────────────────────────────────────────────────
        csv_dir = Path(csv_dir_path)
        all_csv = sorted(csv_dir.glob("*.csv"))
        if all_csv:
            print(f"\n{csv_dir_path}/ 目录下共有 {len(all_csv)} 个 CSV 文件：")
            for f in all_csv:
                print(f"  {f.name}")
        print("\n请输入要导入的文件匹配模式（支持通配符，如 *.csv / 张三*.csv）")
        while True:
            raw_pattern = input("匹配模式（直接回车默认 *.csv）: ").strip() or "*.csv"
            matched = list(csv_dir.glob(raw_pattern))
            if matched:
                print(f"匹配到 {len(matched)} 个文件：{[f.name for f in matched]}")
                break
            print(f"未找到匹配 '{raw_pattern}' 的文件，请重新输入")
        csv_pattern = raw_pattern
        # ─────────────────────────────────────────────────────────────────

        print("\n请选择导入模式:")
        print("1. 全量导入（清空数据库重新导入所有数据）")
        print("2. 增量更新（只导入新增的聊天记录）")

        while True:
            choice = input("请输入选项 (1/2，默认为2): ").strip() or "2"
            if choice in ["1", "2"]:
                break
            print("无效选项，请输入 1 或 2")

        incremental = (choice == "2")

        tracking_data = load_import_tracking(IMPORT_TRACKING_FILE) if incremental else None

        print("\nLoading WeChat CSV files...")
        csv_loader = WeChatCSVLoader(csv_dir_path)
        docs, new_hashes = csv_loader.load(incremental=incremental, tracking_data=tracking_data, csv_pattern=csv_pattern)

        if not docs:
            if incremental:
                print("Info: No new chat records to import")
                return
            else:
                print("Error: No valid chat records found, please check CSV file format")
                return

        print(f"Success: Loaded {len(docs)} {'new' if incremental else 'valid'} chat records")

        print("\nSkipping document splitting (chat records are already atomic units)...")
        splits = docs
        print(f"Using {len(splits)} chat records as-is")

        print("\n请设置并发嵌入线程数（每个线程独立调用 DashScope API）：")
        while True:
            w = input("线程数（直接回车默认 4，推荐 4-8）: ").strip() or "4"
            if w.isdigit() and int(w) >= 1:
                max_workers = int(w)
                break
            print("请输入正整数")

        print("\nCreating/loading vector database...")
        embeddings = DashScopeEmbeddings(model=os.getenv("EMBED_MODEL", "text-embedding-v4"))

        ok = create_vectorstore_with_progress(
            splits, embeddings, collection_name=CHROMA_COLLECTION,
            incremental=incremental, max_workers=max_workers
        )

        if not ok:
            print("Error: Vector database creation failed")
            return

        if incremental and new_hashes:
            tracking_data["imported_hashes"].update(new_hashes)
            save_import_tracking(IMPORT_TRACKING_FILE, tracking_data)
            print(f"已保存 {len(new_hashes)} 条新记录的跟踪信息")

        pm.update_doc_count(persona_id, len(splits))
        print(f"✅ 已更新分身 '{persona['name']}' 的文档数量: {len(splits)}")

        print("\n正在采样消息长度，计算 max_tokens...")
        max_tokens = compute_max_tokens(splits)
        pm.update_model_params(persona_id, {"max_tokens": max_tokens})
        print(f"✅ 已保存 max_tokens={max_tokens} 到分身 '{persona['name']}'")

        print("Success: Vector database ready!")

        print("\nBuilding RAG retrieval chain...")
        vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 20, "fetch_k": 80, "lambda_mult": 0.6},
        )

        prompt = hub.pull("rlm/rag-prompt")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        print("Success: RAG system build complete!")
        interactive_chat(rag_chain)

    except Exception as e:
        print(f"Error: Program execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
