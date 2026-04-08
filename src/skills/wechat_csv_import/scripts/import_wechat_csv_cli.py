"""
微信聊天记录导入脚本 (交互式)
使用方式: python import_wechat_csv.py
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from src.infrastructure.db_client import DBClient
from src.infrastructure.persona_manager import PersonaManager
from src.loaders.csv_loader import WeChatCSVLoader
from src.services.import_service import ImportService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("import_script")


def main(preset_persona_id: str = None):
    print("\n" + "="*50)
    print("   微信聊天记录导入系统 (DigitalTwin-Refactor)")
    print("="*50 + "\n")

    try:
        # 1. 初始化基础设施
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        db_client = DBClient()
        pm = PersonaManager(persist_dir)
        import_service = ImportService(db_client)

        persona = None
        # 2. 选择分身 (Persona)
        if preset_persona_id:
            persona = pm.get(preset_persona_id)
            if persona:
                print(f"已自动锁定分身: {persona['name']}")
        
        if not persona:
            existing_personas = pm.list()
            if existing_personas:
                print("现有分身列表:")
                for i, p in enumerate(existing_personas, 1):
                    print(f"  {i}. {p['name']} (集合: {p['collection']}, 已导入: {p['doc_count']})")
                print(f"  {len(existing_personas) + 1}. 创建新分身")
                
                choice = input(f"\n请选择分身 (1-{len(existing_personas) + 1}) [默认 1]: ").strip() or "1"
                if choice.isdigit():
                    idx = int(choice)
                    if 1 <= idx <= len(existing_personas):
                        persona = existing_personas[idx - 1]
                    elif idx == len(existing_personas) + 1:
                        persona = None
            
            if not persona:
                print("\n--- 创建新分身 ---")
                name = input("请输入分身名称: ").strip()
                while not name:
                    name = input("名称不能为空，请重新输入: ").strip()
                
                print("请输入系统提示词 (描述分身的性格和语气，直接回车使用默认值):")
                system_prompt = input("> ").strip()
                if not system_prompt:
                    system_prompt = "你是一个贴心的数字分身，请模仿聊天记录中的语气回答问题。"
                
                persona = pm.create(name=name, system_prompt=system_prompt)
                print(f"✅ 已创建分身: {persona['name']} (集合: {persona['collection']})")

        collection_name = persona["collection"]
        persona_id = persona["id"]

        # 3. 选择 CSV 文件
        csv_dir = os.getenv("CSV_DIR", "data/csv")
        print(f"\n当前搜索目录: {csv_dir}")
        pattern_input = input("请输入文件匹配模式 (如 *.csv) [默认 *.csv]: ").strip() or "*.csv"
        full_pattern = os.path.join(csv_dir, pattern_input)

        # 4. 选择导入模式
        print("\n选择导入模式:")
        print("  1. 增量更新 (只导入新消息, 推荐)")
        print("  2. 全量导入 (清空并重新导入)")
        mode_choice = input("请选择 (1/2) [默认 1]: ").strip() or "1"
        incremental = (mode_choice == "1")

        if not incremental:
            confirm = input(f"⚠️ 警告: 将清空集合 {collection_name}，确认吗？(y/n): ").lower()
            if confirm == 'y':
                db_client.delete_collection(collection_name)
                print(f"✅ 已清空集合 {collection_name}")
            else:
                print("已取消全量导入。")
                return

        # 5. 执行导入
        tracking_file = os.path.join(persist_dir, f"{collection_name}_tracking.json")
        
        print(f"\n🚀 开始导入任务...")
        result = import_service.import_documents(
            loader_cls=WeChatCSVLoader,
            pattern=full_pattern,
            collection_name=collection_name,
            incremental=incremental,
            tracking_file=tracking_file,
            max_workers=4
        )

        if result.get("status") == "success":
            count = result["count"]
            # 6. 更新分身统计信息
            pm.update_doc_count(persona_id, count)
            
            # 如果是聊天数据，计算并更新 max_tokens
            if result.get("documents"):
                max_tokens = import_service.compute_max_tokens(result["documents"])
                pm.update_model_params(persona_id, {"max_tokens": max_tokens})
            
            print(f"\n✨ 导入成功! 共导入 {count} 条记录。")
        else:
            print(f"\nℹ️ 导入结束: {result.get('status')}")

    except Exception as e:
        logger.error(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
