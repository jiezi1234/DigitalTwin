"""
CSV 文件加载器（用于微信聊天记录）
"""

import os
import csv
import logging
from typing import List
from src.loaders.base import DataLoader
from src.infrastructure.document import Document
from src.infrastructure.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class CSVLoader(DataLoader):
    """CSV 加载器，支持微信聊天记录"""

    def __init__(
        self,
        filepath: str,
        encoding: str = "utf-8",
        message_column: str = "message",
        metadata_columns: List[str] = None,
    ):
        """
        初始化 CSV 加载器

        Args:
            filepath: CSV 文件路径
            encoding: 文件编码
            message_column: 内容列名（默认 "message"）
            metadata_columns: 元数据列列表（如 ["talker", "chat_time", "timestamp"]）
        """
        self.filepath = filepath
        self.encoding = encoding
        self.message_column = message_column
        self.metadata_columns = metadata_columns or ["talker", "chat_time", "timestamp"]

    def load(self) -> List[Document]:
        """加载 CSV 文件"""
        with tracer.start_as_current_span("loader.load") as span:
            span.set_attribute("loader.type", "csv")
            span.set_attribute("loader.filepath", self.filepath)

            documents = []

            try:
                with open(self.filepath, "r", encoding=self.encoding) as f:
                    reader = csv.DictReader(f)

                    for row_num, row in enumerate(reader, start=1):
                        if not row.get(self.message_column):
                            continue

                        # 提取内容
                        content = row[self.message_column].strip()

                        # 提取元数据
                        metadata = {}
                        for col in self.metadata_columns:
                            if col in row:
                                value = row[col].strip()
                                # 尝试将时间戳转换为整数
                                if col in ("chat_time", "timestamp"):
                                    try:
                                        metadata[col] = int(value)
                                    except (ValueError, TypeError):
                                        metadata[col] = value
                                else:
                                    metadata[col] = value

                        # 添加源标识
                        metadata["source"] = "csv"
                        metadata["source_file"] = self.filepath

                        doc = Document(
                            content=content,
                            metadata=metadata,
                        )
                        documents.append(doc)

                logger.info(f"从 {self.filepath} 加载了 {len(documents)} 条记录")
                span.set_attribute("loader.documents_loaded", len(documents))

            except Exception as e:
                logger.error(f"加载 CSV 文件失败: {self.filepath} - {e}")
                span.record_exception(e)
                raise

            return documents


class WeChatCSVLoader(DataLoader):
    """微信聊天记录 CSV 加载器，支持字段清洗和标准化"""

    def __init__(self, filepath: str, encoding: str = "utf-8"):
        self.filepath = filepath
        self.encoding = encoding

    def _should_skip(self, msg: str, type_name: str = "") -> bool:
        """判断是否跳过该消息 (过滤表情、系统提示、过短消息等)"""
        if not msg:
            return True
        msg = msg.strip()
        
        # 基础过滤规则 (同步自 preprocess_csv.py)
        if (len(msg) <= 2 or 
            msg.startswith('[') or 
            msg.startswith('表情') or
            '动画表情' in type_name or
            msg == "I've accepted your friend request. Now let's chat!" or
            '<msg>' in msg):
            return True
        return False

    def load(self) -> List[Document]:
        """加载并解析微信 CSV 文件"""
        with tracer.start_as_current_span("wechat_loader.load") as span:
            span.set_attribute("loader.filepath", self.filepath)
            
            documents = []
            try:
                with open(self.filepath, "r", encoding=self.encoding) as f:
                    reader = csv.DictReader(f)
                    
                    for row in reader:
                        # 必须字段
                        content = row.get("msg") or row.get("message") or ""
                        talker = row.get("talker") or "unknown"
                        is_sender = str(row.get("is_sender", "0"))
                        
                        # 转换 is_sender 为整数 (1=自己, 0=他人)
                        try:
                            is_sender_int = int(is_sender)
                        except ValueError:
                            is_sender_int = 1 if is_sender.lower() in ("true", "self", "1") else 0

                        # 数据清洗
                        type_name = row.get("type_name", "")
                        if self._should_skip(content, type_name):
                            continue

                        # 合并元数据
                        metadata = {
                            "talker": talker,
                            "is_sender": is_sender_int,
                            "chat_time": row.get("CreateTime") or row.get("chat_time") or "",
                            "room": row.get("room_name") or row.get("room") or "",
                            "source": "wechat_csv",
                            "source_file": os.path.basename(self.filepath)
                        }

                        # 格式化内容 (用于向量搜索): "talker: msg" 或 "talker@room: msg"
                        room = metadata["room"]
                        display_name = f"{talker}@{room}" if room else talker
                        formatted_content = f"{display_name}: {content.strip()}"

                        doc = Document(
                            content=formatted_content,
                            metadata=metadata
                        )
                        documents.append(doc)

                logger.info(f"WeChatLoader: 从 {self.filepath} 加载了 {len(documents)} 条有效记录")
                span.set_attribute("loader.documents_loaded", len(documents))

            except Exception as e:
                logger.error(f"WeChatLoader 失败: {self.filepath} - {e}")
                span.record_exception(e)
                raise

            return documents
