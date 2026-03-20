"""
PDF 教材导入脚本 (交互式)
使用方式: python -m src.import_pdf
"""

import os
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from src.infrastructure.db_client import DBClient
from src.infrastructure.persona_manager import PersonaManager
from src.loaders.pdf_loader import PDFLoader
from src.services.import_service import ImportService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("import_pdf")


def main():
    print("\n" + "="*50)
    print("   PDF 教材导入系统 (DigitalTwin-Refactor)")
    print("="*50 + "\n")

    try:
        # 1. 初始化基础设施
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        db_client = DBClient()
        pm = PersonaManager(persist_dir)
        import_service = ImportService(db_client)

        # 2. 选择分身 (Persona)
        existing_personas = pm.list()
        persona = None

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
            
            system_prompt = input("请输入系统提示词 (直接回车使用默认值): ").strip()
            if not system_prompt:
                system_prompt = "你是一个专业的知识库助手，请根据提供的教材内容准确回答问题。"
            
            persona = pm.create(name=name, system_prompt=system_prompt, source_type="knowledge")
            print(f"✅ 已创建分身: {persona['name']} (集合: {persona['collection']})")

        collection_name = persona["collection"]
        persona_id = persona["id"]

        # 3. 配置 PDF 路径
        pdf_dir = os.getenv("PDF_DIR", "data/pdf")
        print(f"\n当前搜索目录: {pdf_dir}")
        pattern_input = input("请输入文件匹配模式 (如 *.pdf) [默认 *.pdf]: ").strip() or "*.pdf"
        full_pattern = os.path.join(pdf_dir, pattern_input)

        # 4. OCR 配置
        ocr_choice = input("是否启用 OCR？(y/n) [默认 y]: ").strip().lower() or "y"
        ocr_enabled = (ocr_choice == "y")

        # 5. 执行导入
        tracking_file = os.path.join(persist_dir, f"{collection_name}_pdf_tracking.json")
        
        print(f"\n🚀 开始导入 PDF 任务 (OCR={ocr_enabled})...")
        result = import_service.import_documents(
            loader_cls=PDFLoader,
            pattern=full_pattern,
            collection_name=collection_name,
            incremental=True,
            tracking_file=tracking_file,
            max_workers=4,
            ocr_enabled=ocr_enabled
        )

        if result.get("status") == "success":
            count = result["count"]
            # 6. 更新分身统计信息 (如果是累加导入，需要获取当前总数)
            stats = db_client.get_stats(collection_name)
            pm.update_doc_count(persona_id, stats.get("total_records", count))
            
            print(f"\n✨ 导入成功! 本次共导入 {count} 条记录。")
            print(f"📚 该分身当前共有 {stats.get('total_records')} 条参考知识。")
        else:
            print(f"\nℹ️ 导入结束: {result.get('status')}")

    except Exception as e:
        logger.error(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
