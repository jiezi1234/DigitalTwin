"""
PDF/文档加载器
使用 PyMuPDF 解析 PDF，按章节/段落智能分块
"""

import os
import re
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# 需要从每页文本中清除的噪音模式
NOISE_PATTERNS = [
    re.compile(r'^Principle and Technology of Database\s*$', re.MULTILINE),
    re.compile(r'^NOTES\s*$', re.MULTILINE),
    re.compile(r'^Copyright\s*©.*$', re.MULTILINE),
    re.compile(r'^Page\s+\d+\s*$', re.MULTILINE),
]


@dataclass
class TextChunk:
    """文本块数据模型"""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PDFLoader:
    """PDF 文档加载器"""

    # 常见章节标题模式（中文教材）
    CHAPTER_PATTERNS = [
        re.compile(r'^第[一二三四五六七八九十\d]+章\s+'),
        re.compile(r'^第[一二三四五六七八九十\d]+节\s+'),
        re.compile(r'^\d+\.\d+(\.\d+)?\s+'),  # 1.1 / 1.1.1 格式
        re.compile(r'^Chapter\s+\d+', re.IGNORECASE),
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        extra_noise_patterns: List[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 允许用户传入额外的噪音正则
        self.noise_patterns = list(NOISE_PATTERNS)
        if extra_noise_patterns:
            for p in extra_noise_patterns:
                self.noise_patterns.append(re.compile(p, re.MULTILINE))

    @staticmethod
    def clean_page_text(text: str, patterns: List[re.Pattern] = None) -> str:
        """清除页面中的重复噪音信息（标题、版权、页码等）"""
        if patterns is None:
            patterns = NOISE_PATTERNS
        for pattern in patterns:
            text = pattern.sub('', text)
        # 清理多余空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def load_pdf(self, file_path: str) -> List[TextChunk]:
        """
        加载 PDF 文件并分块

        Args:
            file_path: PDF 文件路径

        Returns:
            TextChunk 列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF 文件不存在: {file_path}")

        logger.info("正在加载 PDF: %s", file_path)
        doc = fitz.open(file_path)
        filename = os.path.basename(file_path)

        # 1. 逐页提取文本，记录页码
        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                # 清除每页的噪音信息（标题、版权、页码等）
                cleaned = self.clean_page_text(text, self.noise_patterns)
                if cleaned:
                    pages.append({
                        "text": cleaned,
                        "page": page_num + 1,  # 1-indexed
                    })

        doc.close()
        logger.info("PDF 共 %d 页，有效页 %d 页", page_num + 1, len(pages))

        # 2. 检测章节结构
        pages_with_chapters = self._detect_chapters(pages)

        # 3. 分块
        chunks = self._split_into_chunks(pages_with_chapters, filename)
        logger.info("生成 %d 个文本块", len(chunks))

        return chunks

    def load_directory(self, dir_path: str) -> List[TextChunk]:
        """
        加载目录下所有 PDF 文件

        Args:
            dir_path: 目录路径

        Returns:
            所有文件的 TextChunk 列表
        """
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"目录不存在: {dir_path}")

        all_chunks = []
        pdf_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.pdf')]

        if not pdf_files:
            logger.warning("目录 %s 下没有找到 PDF 文件", dir_path)
            return all_chunks

        for pdf_file in sorted(pdf_files):
            file_path = os.path.join(dir_path, pdf_file)
            try:
                chunks = self.load_pdf(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error("加载 %s 失败: %s", pdf_file, e)

        logger.info("共加载 %d 个 PDF 文件，%d 个文本块", len(pdf_files), len(all_chunks))
        return all_chunks

    def _detect_chapters(self, pages: List[Dict]) -> List[Dict]:
        """检测并标注章节信息"""
        current_chapter = "前言"
        current_section = ""

        for page_info in pages:
            lines = page_info["text"].split("\n")
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                for pattern in self.CHAPTER_PATTERNS:
                    if pattern.match(line_stripped):
                        if '章' in line_stripped or line_stripped.startswith('Chapter'):
                            current_chapter = line_stripped
                            current_section = ""
                        elif '节' in line_stripped or re.match(r'^\d+\.\d+', line_stripped):
                            current_section = line_stripped
                        break

            page_info["chapter"] = current_chapter
            page_info["section"] = current_section

        return pages

    def _split_into_chunks(self, pages: List[Dict], filename: str) -> List[TextChunk]:
        """将页面文本分块"""
        chunks = []

        for page_info in pages:
            text = page_info["text"].strip()
            if not text:
                continue

            page = page_info["page"]
            chapter = page_info["chapter"]
            section = page_info["section"]

            # 按段落先分割
            paragraphs = re.split(r'\n{2,}', text)
            current_chunk = ""

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # 检查是否有新的章节标题
                for pattern in self.CHAPTER_PATTERNS:
                    if pattern.match(para):
                        if '章' in para or para.startswith('Chapter'):
                            chapter = para
                            section = ""
                        elif '节' in para or re.match(r'^\d+\.\d+', para):
                            section = para
                        break

                if len(current_chunk) + len(para) + 1 > self.chunk_size:
                    # 当前块已满，保存并开始新块
                    if current_chunk:
                        chunks.append(TextChunk(
                            text=current_chunk,
                            metadata={
                                "source": filename,
                                "page": page,
                                "chapter": chapter,
                                "section": section,
                                "type": "textbook",
                            }
                        ))
                    # 开始新块（带重叠）
                    if len(current_chunk) > self.chunk_overlap:
                        overlap = current_chunk[-self.chunk_overlap:]
                        current_chunk = overlap + "\n" + para
                    else:
                        current_chunk = para
                else:
                    current_chunk = (current_chunk + "\n" + para) if current_chunk else para

            # 保存最后一个块
            if current_chunk.strip():
                chunks.append(TextChunk(
                    text=current_chunk,
                    metadata={
                        "source": filename,
                        "page": page,
                        "chapter": chapter,
                        "section": section,
                        "type": "textbook",
                    }
                ))

        return chunks
