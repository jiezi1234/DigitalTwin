"""
增量导入跟踪工具
"""

import hashlib
import json
import os


def generate_record_hash(source: str, row, create_time: str, msg: str) -> str:
    """生成聊天记录的唯一哈希值（含文件名+行号防止同内容误判）"""
    unique_str = f"{source}|{row}|{create_time}|{msg}"
    return hashlib.sha256(unique_str.encode('utf-8')).hexdigest()


def load_import_tracking(tracking_file: str) -> dict:
    """加载已导入记录的跟踪信息"""
    if os.path.exists(tracking_file):
        try:
            with open(tracking_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"警告: 无法加载导入跟踪文件: {e}")
    return {"imported_hashes": set(), "file_timestamps": {}}


def save_import_tracking(tracking_file: str, tracking_data: dict):
    """保存导入跟踪信息"""
    try:
        os.makedirs(os.path.dirname(tracking_file) or ".", exist_ok=True)
        save_data = {
            "imported_hashes": list(tracking_data["imported_hashes"]),
            "file_timestamps": tracking_data["file_timestamps"],
        }
        with open(tracking_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"警告: 无法保存导入跟踪文件: {e}")
