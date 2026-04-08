import os
import sys
import time
import signal
import subprocess
import glob
from dotenv import load_dotenv

load_dotenv()

from src.infrastructure.persona_manager import PersonaManager
from src.api.config import Config
from src.skills.wechat_csv_import.scripts.import_wechat_csv_cli import main as import_wechat_csv

pm = PersonaManager(Config.CHROMA_PERSIST_DIR)

def print_header():
    print("\n" + "="*50)
    print("   [Digital Twin Command Center]")
    print("="*50 + "\n")

def menu_create_persona():
    print("\n--- 创建新数字分身 ---")
    name = input("请输入分身名称: ").strip()
    if not name:
        print("操作已取消。")
        return

    print("请选择分身类型：")
    print("  [1] 聊天型 (学习对话语气, 日常对话)")
    print("  [2] 知识型 (基于专业教材查询, 严谨风格)")
    type_choice = input("请选择 (1/2) [默认 1]: ").strip() or "1"
    
    source_type = "knowledge" if type_choice == "2" else "chat"
    
    print("\n请输入系统提示词 (Prompt)，直接回车将使用默认预设:")
    system_prompt = input("> ").strip()
    if not system_prompt:
        if source_type == "knowledge":
            system_prompt = "你是一个专业的知识分身，请基于知识库给出严谨负责的解答。"
        else:
            system_prompt = "你是一个贴心的数字分身，请模仿记录中的语气回答问题。"
            
    persona = pm.create(name=name, system_prompt=system_prompt, source_type=source_type)
    print(f"\n✅ 成功建档！分身「{persona['name']}」(类型: {source_type}) 现已就绪。")

def menu_feed_data():
    existing = pm.list()
        
    print("\n--- 喂给分身资料 ---")
    print("请先选择要把资料喂给哪个目标身份库:")
    print("  [0] 内置基础数字助教 (Tutor)  -> 专供喂送 PDF 教材与课件")
    for i, p in enumerate(existing, 1):
        ptype = "知识型" if p.get("source_type") == "knowledge" else "聊天型"
        print(f"  [{i}] {p['name']} ({ptype}, 已有记录 {p['doc_count']} 条)")
    
    choice = input(f"请输入序号 (0-{len(existing)}): ").strip()
    if not choice.isdigit() or not (0 <= int(choice) <= len(existing)):
        print("选择无效，操作取消。")
        return
        
    if choice == "0":
        pdf_dir = os.path.join(Config.PROJECT_ROOT, "data", "pdf")
        all_pdfs = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
        if not all_pdfs:
            print(f"未在 {pdf_dir} 发现 PDF 文件，操作取消。")
            return

        print("\n检测到以下可导入 PDF：")
        for idx, path in enumerate(all_pdfs, 1):
            print(f"  [{idx}] {os.path.basename(path)}")

        selected = input("请选择要导入的文件序号（逗号分隔，默认 all）: ").strip().lower()
        if not selected or selected == "all":
            selected_paths = list(all_pdfs)
        else:
            indices = []
            for token in selected.split(","):
                token = token.strip()
                if token.isdigit():
                    i = int(token)
                    if 1 <= i <= len(all_pdfs):
                        indices.append(i - 1)
            indices = sorted(set(indices))
            if not indices:
                print("未选择有效文件，操作取消。")
                return
            selected_paths = [all_pdfs[i] for i in indices]

        textbook_candidates = [p for p in selected_paths if "textbook" in os.path.basename(p).lower()]
        textbook_file = textbook_candidates[0] if textbook_candidates else None
        notes_files = [p for p in selected_paths if p != textbook_file]

        if selected_paths and not textbook_file:
            print("\n当前未包含 textbook.pdf，将按“仅导入 notes”模式继续。")
            print("如果你希望本次同时导入 textbook，请输入其序号；直接回车则跳过。")
            manual = input("可选：指定 textbook 序号（回车跳过）: ").strip()
            if manual.isdigit():
                m = int(manual)
                if 1 <= m <= len(all_pdfs):
                    textbook_file = all_pdfs[m - 1]
                    notes_files = [p for p in selected_paths if p != textbook_file]

        print("\n请选择导入模式：")
        print("  [1] 增量导入 (推荐，仅导入新内容)")
        print("  [2] 全量重建 (清空后重建)")
        mode = input("请选择 (1/2) [默认 1]: ").strip() or "1"
        reset = (mode == "2")

        cmd = [
            sys.executable,
            "-m",
            "src.skills.course_material_import.scripts.import_course_materials_cli",
            "--persist-dir", os.path.join(Config.PROJECT_ROOT, "chroma_db_mm"),
            "--output-root", os.path.join(Config.PROJECT_ROOT, "output", "course_mm"),
            "--notes-files",
        ]
        cmd.extend(notes_files)
        if textbook_file:
            cmd.extend(["--textbook-file", textbook_file])
        else:
            cmd.append("--skip-textbook")
        if reset:
            cmd.append("--reset")

        print(f"\n[系统] 已另起终端执行教材导入 (目标: 内置助教, 模式: {'全量重建' if reset else '增量导入'})...")
        try:
            CREATE_NEW_CONSOLE = getattr(subprocess, 'CREATE_NEW_CONSOLE', 0x00000010)
            subprocess.Popen(cmd, creationflags=CREATE_NEW_CONSOLE)
        except Exception as e:
            print(f"独立终端启动失败 ({e})，退回当前终端执行...")
            subprocess.run(cmd, check=False)
            
    else:
        target_persona = existing[int(choice) - 1]
        print(f"\n[系统] 已另起终端牵引微信录入管线 (目标: {target_persona['name']})...")
        code = f"from src.skills.wechat_csv_import.scripts.import_wechat_csv_cli import main; main('{target_persona['id']}'); input('\\n[提示] 按回车键(Enter)关闭本导入串口...')"
        try:
            CREATE_NEW_CONSOLE = getattr(subprocess, 'CREATE_NEW_CONSOLE', 0x00000010)
            subprocess.Popen([sys.executable, "-c", code], creationflags=CREATE_NEW_CONSOLE)
        except Exception as e:
            print(f"独立终端启动失败 ({e}), 退回静默线程...")
            from src.skills.wechat_csv_import.scripts.import_wechat_csv_cli import main as import_wechat_csv
            import_wechat_csv(preset_persona_id=target_persona["id"])

def menu_launch_chat():
    existing = pm.list()
    
    print("\n--- 开机进行对话 ---")
    print("系统可用分身列表:")
    print(f"  [0] 内置基础数字助教 (Tutor)")
    for i, p in enumerate(existing, 1):
        ptype = "知识型" if p.get("source_type") == "knowledge" else "聊天型"
        print(f"  [{i}] {p['name']} ({ptype})")
        
    choice = input(f"你要打开哪个空间的通讯端? (0-{len(existing)}): ").strip()
    if not choice.isdigit() or not (0 <= int(choice) <= len(existing)):
        print("选择无效。")
        return
    
    target_url = f"http://localhost:{Config.PORT}"
    
    if choice == "0":
        target_url = f"{target_url}/tutor"
        print(f"\n🌟 专属通道已连接 => 请浏览器访问: {target_url}\n")
    else:
        # 指向原生入口
        pname = existing[int(choice)-1]['name']
        print(f"\n🌟 专属通道 ({pname}) 已连接 => 请浏览器访问: {target_url}\n")
        
    print("-" * 50)
    print("已为你另起终端开启后台通讯服务！")
    print("本控制台已释放，您可以继续在此执行其它操作。")
    print("-" * 50)
    
    # 在新终端窗口拉起 run_server（仅支持 Windows）
    try:
        CREATE_NEW_CONSOLE = getattr(subprocess, 'CREATE_NEW_CONSOLE', 0x00000010)
        # 用当前激活环境的 Python 解释器去启动
        subprocess.Popen([sys.executable, "-m", "src.run_server"], creationflags=CREATE_NEW_CONSOLE)
    except Exception as e:
        print(f"启动独立终端失败: {e}，将尝试在后台静默运行...")
        subprocess.Popen([sys.executable, "-m", "src.run_server"])

def main_loop():
    while True:
        print_header()
        print("请选择你要执行的操作：")
        print("[ 1 ] 管理/创建新数字分身")
        print("[ 2 ] 喂给分身资料 (数据注入系统)")
        print("[ 3 ] 和指定的分身对话 (启动后台服与专属入口)")
        print("[ x ] 退出系统")
        print("="*50)
        
        opt = input("Command> ").strip().lower()
        if opt in ('x', 'q', 'exit'):
            print("再见！Digital Twin 系统已关闭。")
            sys.exit(0)
            
        elif opt == '1':
            menu_create_persona()
        elif opt == '2':
            menu_feed_data()
        elif opt == '3':
            menu_launch_chat()
            # 退出聊天室后清屏或稍微等待
            time.sleep(1)
        else:
            print("未知命令，请重新输入。")

if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\n再见！Digital Twin 系统已关闭。")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    main_loop()
