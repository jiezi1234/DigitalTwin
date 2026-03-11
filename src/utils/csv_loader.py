"""
微信聊天记录 CSV 加载器

支持多种数据格式，自动检测并统一处理
"""

import os
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Set, Tuple, Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="Processing", total=None):
        print(f"{desc}...")
        return iterable

from langchain_core.documents import Document
from ..models.chat_record_model import ChatRecord, ChatRecordSchema


class WeChatCSVLoader:
    """自定义微信聊天记录CSV加载器

    支持多种数据格式：
    - WeChat标准格式：包含 MsgSvrID, type_name, room_name 等完整字段
    - 简洁格式：仅包含 talker, msg, is_sender 等基本字段
    """

    def __init__(self, csv_folder_path, encoding="utf-8"):
        self.csv_folder_path = Path(csv_folder_path)
        self.encoding = encoding

    def _parse_csv_file(self, csv_file: Path) -> Tuple[str, List[ChatRecord], List[str]]:
        """解析单个CSV文件

        Returns:
            (format_type, records, raw_rows)
        """
        records = []
        raw_rows = []

        with open(csv_file, 'r', encoding=self.encoding) as f:
            # 检测格式
            sample_lines = []
            f.seek(0)
            for _ in range(3):
                sample_lines.append(f.readline())
            f.seek(0)

            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("无法读取CSV文件头")

            format_type = ChatRecordSchema.detect_format(reader.fieldnames)
            if format_type == ChatRecordSchema.FORMAT_UNKNOWN:
                raise ValueError(
                    f"无法识别数据格式。检测到的字段: {reader.fieldnames}\n"
                    f"必须包含以下之一：\n"
                    f"1. 完整格式: id, MsgSvrID, type_name, is_sender, talker, msg, CreateTime\n"
                    f"2. 简洁格式: is_sender, talker, msg"
                )

            # 逐行解析
            for row in reader:
                raw_rows.append(str(row))
                try:
                    record = ChatRecord.from_dict(row)
                    records.append(record)
                except ValueError as e:
                    print(f"    ⚠️ 跳过无效行: {e}")
                    continue

        return format_type, records, raw_rows

    def _should_skip_message(self, record: ChatRecord) -> bool:
        """判断是否应该跳过此消息"""
        msg = record.msg.strip()

        # 过滤无效消息
        if (len(msg) <= 2 or
            msg.startswith('[') or
            msg.startswith('表情') or
            '动画表情' in (record.type_name or '') or
            msg == "I've accepted your friend request. Now let's chat!" or
            '<msg>' in msg):
            return True

        return False

    def _record_to_document(self, record: ChatRecord, csv_filename: str) -> Document:
        """将ChatRecord转换为LangChain Document"""

        # 格式化消息内容
        location = f"@{record.room_name}" if record.has_room() else ""
        formatted_content = f"{record.talker}{location}: {record.msg}"

        # 解析时间戳
        chat_timestamp = 0
        if record.has_time():
            try:
                chat_timestamp = int(datetime.fromisoformat(record.CreateTime).timestamp())
            except Exception:
                try:
                    chat_timestamp = int(datetime.strptime(record.CreateTime, "%Y-%m-%d %H:%M:%S").timestamp())
                except Exception:
                    chat_timestamp = 0

        # 构建Document的metadata
        metadata = {
            "source": csv_filename,
            "chat_time": chat_timestamp,
            "chat_time_str": record.CreateTime or "",
            "sender": record.talker,
            "msg_type": record.msg_type(),
            "room": record.room_name or "",
            "is_sender": record.is_sender,
            "msg_content": record.msg[:200],
        }

        # 添加可选字段
        if record.MsgSvrID:
            metadata["MsgSvrID"] = record.MsgSvrID
        if record.is_forward is not None:
            metadata["is_forward"] = record.is_forward

        return Document(page_content=formatted_content, metadata=metadata)

    def load(self, incremental=False, tracking_data=None, csv_pattern: str = None) -> Tuple[List[Document], Set[str]]:
        """加载所有CSV文件并返回文档列表

        Args:
            incremental: 是否为增量更新模式
            tracking_data: 导入跟踪数据（增量模式下使用）
            csv_pattern: glob 匹配模式，覆盖 CSV_FILE_PATTERN 环境变量

        Returns:
            (documents, new_hashes)
        """
        from .tracking import generate_record_hash

        documents = []
        new_hashes = set()
        skipped_count = 0

        if tracking_data is None:
            tracking_data = {"imported_hashes": set(), "file_timestamps": {}}
        else:
            tracking_data["imported_hashes"] = set(tracking_data.get("imported_hashes", []))

        pattern = csv_pattern or os.getenv("CSV_FILE_PATTERN", "*.csv")
        csv_files = list(self.csv_folder_path.glob(pattern))
        print(f"找到 {len(csv_files)} 个CSV文件（匹配: {pattern}）")

        for csv_file in tqdm(csv_files, desc="处理CSV文件"):
            print(f"\n正在处理: {csv_file.name}")

            try:
                format_type, records, _ = self._parse_csv_file(csv_file)
                print(f"  检测到格式: {format_type}")

                processed_count = 0
                valid_count = 0
                skipped_format_count = 0

                for row_idx, record in enumerate(records, start=2):  # 从第2行开始（第1行是表头）
                    try:
                        # 检查消息是否应该被跳过
                        if self._should_skip_message(record):
                            skipped_format_count += 1
                            continue

                        # 增量导入：检查是否已经导入
                        if incremental:
                            record_hash = generate_record_hash(
                                str(csv_file.name), row_idx,
                                record.CreateTime or "", record.msg
                            )
                            if record_hash in tracking_data["imported_hashes"]:
                                skipped_count += 1
                                continue
                            new_hashes.add(record_hash)

                        # 转换为Document并添加到列表
                        doc = self._record_to_document(record, csv_file.name)
                        documents.append(doc)
                        valid_count += 1

                    except Exception as e:
                        print(f"    ⚠️ 第 {row_idx} 行处理失败: {e}")
                        continue

                    processed_count += 1

                print(f"  - 处理了 {processed_count} 条记录，有效 {valid_count} 条，过滤 {skipped_format_count} 条")

            except Exception as e:
                print(f"  ❌ 处理文件时出错: {e}")
                continue

        if incremental and skipped_count > 0:
            print(f"\n增量更新: 跳过 {skipped_count} 条已导入的记录")

        return documents, new_hashes
