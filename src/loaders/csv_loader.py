"""
CSV 文件加载器（用于微信聊天记录）
"""

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
