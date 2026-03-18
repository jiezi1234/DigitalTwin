"""
PDF 文件加载器（用于教材）
"""

import logging
from typing import List, Optional
from src.loaders.base import DataLoader
from src.infrastructure.document import Document
from src.infrastructure.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class PDFLoader(DataLoader):
    """PDF 加载器，支持教材和文档"""

    def __init__(
        self,
        filepath: str,
        extract_metadata: bool = True,
    ):
        """
        初始化 PDF 加载器

        Args:
            filepath: PDF 文件路径
            extract_metadata: 是否提取页码等元数据
        """
        self.filepath = filepath
        self.extract_metadata = extract_metadata

    def load(self) -> List[Document]:
        """加载 PDF 文件"""
        with tracer.start_as_current_span("loader.load") as span:
            span.set_attribute("loader.type", "pdf")
            span.set_attribute("loader.filepath", self.filepath)

            documents = []

            try:
                # 懒加载 PyPDF2 以避免硬依赖
                from PyPDF2 import PdfReader

                reader = PdfReader(self.filepath)

                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()

                    if not text.strip():
                        continue

                    metadata = {
                        "source": "pdf",
                        "source_file": self.filepath,
                        "page": page_num + 1,
                    }

                    doc = Document(
                        content=text,
                        metadata=metadata,
                    )
                    documents.append(doc)

                logger.info(f"从 {self.filepath} 加载了 {len(documents)} 页")
                span.set_attribute("loader.documents_loaded", len(documents))

            except ImportError:
                logger.error("PyPDF2 未安装，请运行: pip install PyPDF2")
                span.record_exception(ImportError("PyPDF2 not installed"))
                raise

            except Exception as e:
                logger.error(f"加载 PDF 文件失败: {self.filepath} - {e}")
                span.record_exception(e)
                raise

            return documents
