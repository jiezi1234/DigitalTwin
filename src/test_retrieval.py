import os
import chromadb
from dotenv import load_dotenv
import dashscope

load_dotenv()

def main():
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name = "texttest_embeddings"
    api_key = os.getenv("DASHSCOPE_API_KEY")
    dashscope.api_key = api_key

    client = chromadb.PersistentClient(path=persist_dir)
    try:
        collection = client.get_collection(name=collection_name)
        print(f"成功连接集合: {collection_name}, 当前共有 {collection.count()} 条记录\n")
    except Exception as e:
        print(f"连接集合失败: {e}")
        return

    # 定义测试问题
    test_queries = [
        "数据库系统的基础理论包括哪些？",  # 对应前5页（教材内容摘要）
        "Introduction to Database System",  # 对应后5页（Notes标题/内容）
    ]

    for query in test_queries:
        print(f"🔍 检索问题: '{query}'")
        
        # 1. 生成查询向量
        resp = dashscope.TextEmbedding.call(
            model=os.getenv("EMBED_MODEL", "text-embedding-v4"),
            input=query,
            text_type="query"
        )
        if resp.status_code != 200:
            print(f"错误: {resp.message}")
            continue
        query_vector = resp.output["embeddings"][0]["embedding"]

        # 2. 检索最相似的 2 个块
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=2
        )

        # 3. 打印结果
        for i in range(len(results["documents"][0])):
            doc = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i]
            print(f"  [{i+1}] 相似度(距离): {dist:.4f}")
            print(f"      来源: {meta.get('source')} | 页码: {meta.get('page')}")
            print(f"      OCR: {meta.get('ocr')} | 章节: {meta.get('chapter')}")
            snippet = doc.replace('\n', ' ')[:100] + "..."
            print(f"      内容摘要: {snippet}")
            print("-" * 30)
        print("\n")

if __name__ == "__main__":
    main()
