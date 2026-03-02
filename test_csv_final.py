# 数据导入脚本，使用 ChromaDB 本地向量数据库（无需安装任何数据库服务）
import getpass
import os
import glob
from pathlib import Path
import time
import json
import hashlib
try:
    from tqdm import tqdm
except ImportError:
    # 如果没有tqdm，使用简单的替代
    def tqdm(iterable, desc="Processing", total=None):
        print(f"{desc}...")
        return iterable

# 优先使用环境变量，缺失时再交互式输入
os.environ["DASHSCOPE_API_KEY"] = "sk-bae62c151c524da4b4ee5f04e4e19a3f"
import dashscope
dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

from langchain_community.chat_models.tongyi import ChatTongyi

llm = ChatTongyi(model="qwen-plus")

import bs4
from langchain import hub
# 导入 ChromaDB 向量数据库
from langchain_chroma import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ChromaDB 本地持久化目录和集合名称
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "wechat_embeddings")

# ========== 增量更新相关功能 ==========

IMPORT_TRACKING_FILE = "import_tracking.json"

def load_import_tracking():
    """加载已导入记录的跟踪信息"""
    if os.path.exists(IMPORT_TRACKING_FILE):
        try:
            with open(IMPORT_TRACKING_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"警告: 无法加载导入跟踪文件: {e}")
            return {"imported_hashes": set(), "file_timestamps": {}}
    return {"imported_hashes": set(), "file_timestamps": {}}

def save_import_tracking(tracking_data):
    """保存导入跟踪信息"""
    try:
        # 将set转换为list以便JSON序列化
        save_data = {
            "imported_hashes": list(tracking_data["imported_hashes"]),
            "file_timestamps": tracking_data["file_timestamps"]
        }
        with open(IMPORT_TRACKING_FILE, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"警告: 无法保存导入跟踪文件: {e}")

def generate_record_hash(chat_data):
    """生成聊天记录的唯一哈希值用于去重"""
    # 使用时间、发送者、内容、消息类型组合生成唯一标识
    unique_str = f"{chat_data.get('CreateTime', '')}{chat_data.get('talker', '')}{chat_data.get('msg', '')}{chat_data.get('type_name', '')}"
    return hashlib.md5(unique_str.encode('utf-8')).hexdigest()

# ========================================

class WeChatCSVLoader:
    """自定义微信聊天记录CSV加载器"""

    def __init__(self, csv_folder_path, encoding="utf-8"):
        self.csv_folder_path = Path(csv_folder_path)
        self.encoding = encoding

    def load(self, incremental=False, tracking_data=None):
        """加载所有CSV文件并返回文档列表

        Args:
            incremental: 是否为增量更新模式
            tracking_data: 导入跟踪数据（增量模式下使用）
        """
        documents = []
        new_hashes = set()
        skipped_count = 0

        # 初始化跟踪数据
        if tracking_data is None:
            tracking_data = {"imported_hashes": set(), "file_timestamps": {}}
        else:
            # 将list转换为set以便快速查找
            tracking_data["imported_hashes"] = set(tracking_data.get("imported_hashes", []))

        # 查找所有CSV文件
        csv_files = list(self.csv_folder_path.glob("zhenhuan_chat.csv"))
        print(f"找到 {len(csv_files)} 个CSV文件")

        for csv_file in tqdm(csv_files, desc="处理CSV文件"):
            print(f"正在处理: {csv_file.name}")

            # 使用CSVLoader加载单个文件
            loader = CSVLoader(
                file_path=str(csv_file),
                encoding=self.encoding,
                csv_args={'delimiter': ','}
            )

            try:
                file_docs = loader.load()
                processed_count = 0
                valid_count = 0

                # 处理每个文档
                for doc in file_docs:
                    try:
                        # 解析CSV行数据
                        content_parts = doc.page_content.split('\n')
                        chat_data = {}

                        for part in content_parts:
                            if ':' in part:
                                key, value = part.split(':', 1)
                                chat_data[key.strip()] = value.strip()

                        # 检查是否有有效的消息内容
                        msg_content = chat_data.get('msg', '').strip()
                        if not msg_content:
                            continue

                        # 增量更新模式下检查是否已导入
                        if incremental:
                            record_hash = generate_record_hash(chat_data)
                            if record_hash in tracking_data["imported_hashes"]:
                                skipped_count += 1
                                continue
                            new_hashes.add(record_hash)

                        # 过滤无意义消息
                        if (len(msg_content) <= 2 or
                            msg_content.startswith('[') or
                            msg_content.startswith('表情') or
                            '动画表情' in chat_data.get('type_name', '') or
                            msg_content == "I've accepted your friend request. Now let's chat!" or
                            '<msg>' in msg_content):  # 过滤XML格式的系统消息
                            continue

                        # 格式化聊天内容
                        formatted_content = f"""聊天记录:
时间: {chat_data.get('CreateTime', '未知时间')}
发送者: {chat_data.get('talker', '未知用户')}
消息类型: {chat_data.get('type_name', '文本')}
内容: {msg_content}
房间: {chat_data.get('room_name', '私聊')}
是否自己发送: {'是' if chat_data.get('is_sender') == '1' else '否'}"""

                        # 创建新文档
                        new_doc = Document(
                            page_content=formatted_content,
                            metadata={
                                "source": str(csv_file.name),
                                "chat_time": chat_data.get('CreateTime', ''),
                                "sender": chat_data.get('talker', ''),
                                "msg_type": chat_data.get('type_name', ''),
                                "room": chat_data.get('room_name', ''),
                                "is_sender": chat_data.get('is_sender', '0'),
                                "msg_content": msg_content[:200]  # 截取前200字符用于检索
                            }
                        )
                        documents.append(new_doc)
                        valid_count += 1

                    except Exception as e:
                        continue  # 跳过有问题的记录

                    processed_count += 1

                print(f"  - 处理了 {processed_count} 条记录，有效记录 {valid_count} 条")

            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {e}")
                continue

        if incremental and skipped_count > 0:
            print(f"\n增量更新: 跳过 {skipped_count} 条已导入的记录")

        return documents, new_hashes

def create_vectorstore_with_progress(documents, embeddings, batch_size=100, incremental=False):
    """分批创建/更新向量数据库，使用 ChromaDB 本地存储

    Args:
        documents: 要导入的文档列表
        embeddings: embedding模型
        batch_size: 批次大小（ChromaDB 批量添加推荐 100~500）
        incremental: 是否为增量更新模式（True=追加，False=清空重建）
    """
    print(f"开始{'增量更新' if incremental else '全量导入'}向量数据库，共 {len(documents)} 个文档...")
    print(f"ChromaDB 持久化目录: {CHROMA_PERSIST_DIR}")
    print(f"集合名称: {CHROMA_COLLECTION}")

    try:
        if not incremental:
            # 全量模式：先删除旧集合，再重建
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            try:
                client.delete_collection(CHROMA_COLLECTION)
                print(f"✅ 已清空旧集合 '{CHROMA_COLLECTION}'")
            except Exception:
                pass  # 集合不存在时忽略

        vectorstore = None
        failed_batches = 0
        max_retries = 3
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(documents), batch_size), desc="写入向量数据库", total=total_batches):
            batch = documents[i:i + batch_size]

            retry_count = 0
            while retry_count < max_retries:
                try:
                    if vectorstore is None:
                        # 第一批：初始化 Chroma 并写入
                        vectorstore = Chroma.from_documents(
                            documents=batch,
                            embedding=embeddings,
                            collection_name=CHROMA_COLLECTION,
                            persist_directory=CHROMA_PERSIST_DIR,
                        )
                        print(f"✅ 向量数据库初始化成功，第一批写入 {len(batch)} 个文档")
                    else:
                        # 后续批次：追加写入
                        vectorstore.add_documents(batch)

                    break  # 成功，退出重试

                except Exception as e:
                    retry_count += 1
                    print(f"批次 {i//batch_size + 1} 处理失败 (重试 {retry_count}/{max_retries}): {e}")
                    if retry_count >= max_retries:
                        failed_batches += 1
                        print(f"批次 {i//batch_size + 1} 最终失败，跳过")
                        break
                    time.sleep(2 ** retry_count)  # 指数退避

            # 每20批稍作休息，避免 API 限流
            if (i // batch_size + 1) % 20 == 0:
                time.sleep(1)
            else:
                time.sleep(0.1)

        if failed_batches > 0:
            print(f"⚠️ 警告: {failed_batches} 个批次处理失败")

        return vectorstore

    except Exception as e:
        print(f"❌ 向量数据库创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

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

    # 先做一个测试查询
    test_queries = [
        "聊天记录中都有哪些人参与了对话？",
        "最近在聊什么话题？"
    ]

    for test_query in test_queries:
        print(f"\nTest query: {test_query}")
        try:
            result = rag_chain.invoke(test_query)
            print(f"Result: {result}")
            break  # 成功一个就够了
        except Exception as e:
            print(f"Error: Test query failed: {e}")
            continue

    # 交互式查询
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
            end_time = time.time()

            print(f"Answer: {result}")
            print(f"Time: {end_time - start_time:.2f} seconds")

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

        # 检查CSV文件夹
        if not os.path.exists("csv"):
            print("Error: csv folder not found, please ensure CSV files are in csv directory")
            return

        # 询问用户是全量导入还是增量更新
        print("\n请选择导入模式:")
        print("1. 全量导入（清空数据库重新导入所有数据）")
        print("2. 增量更新（只导入新增的聊天记录）")

        while True:
            choice = input("请输入选项 (1/2，默认为2): ").strip() or "2"
            if choice in ["1", "2"]:
                break
            print("无效选项，请输入 1 或 2")

        incremental = (choice == "2")

        # 加载导入跟踪数据
        tracking_data = load_import_tracking() if incremental else None

        # 加载微信聊天记录CSV数据
        print("\nLoading WeChat CSV files...")
        csv_loader = WeChatCSVLoader("csv")
        docs, new_hashes = csv_loader.load(incremental=incremental, tracking_data=tracking_data)

        if not docs:
            if incremental:
                print("Info: No new chat records to import")
                return
            else:
                print("Error: No valid chat records found, please check CSV file format")
                return

        print(f"Success: Loaded {len(docs)} {'new' if incremental else 'valid'} chat records")


        # 对于聊天记录，每条已经是独立完整的单元，跳过文本分割避免重复
        print("\nSkipping document splitting (chat records are already atomic units)...")
        splits = docs  # 直接使用原始文档，不进行分割
        print(f"Using {len(splits)} chat records as-is")

        # 创建向量数据库
        print("\nCreating/loading vector database...")
        embeddings = DashScopeEmbeddings(model="text-embedding-v3")

        # 【修改】调用更新后的函数，传入 incremental 参数
        vectorstore = create_vectorstore_with_progress(
            splits,
            embeddings,
            batch_size=200,
            incremental=incremental
        )

        if vectorstore is None:
            print("Error: Vector database creation failed")
            return

        # 保存导入跟踪数据
        if incremental and new_hashes:
            tracking_data["imported_hashes"].update(new_hashes)
            save_import_tracking(tracking_data)
            print(f"已保存 {len(new_hashes)} 条新记录的跟踪信息")

        print("Success: Vector database ready!")

        # 构建RAG链
        print("\nBuilding RAG retrieval chain...")
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 30}  # 检索30个最相关的片段
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

        # 开始交互式对话
        interactive_chat(rag_chain)

    except Exception as e:
        print(f"Error: Program execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()