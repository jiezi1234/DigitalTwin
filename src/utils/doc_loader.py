"""
PDF/文档加载器
使用 PyMuPDF 解析 PDF，按章节/段落智能分块
对图片型页面（扫描件）自动启用 Tesseract OCR
"""

import os
import re
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# 每页图片型判断阈值：页面文字字符数低于此值则视为图片页面，启用 OCR
OCR_TEXT_THRESHOLD = 50

# OCR 语言配置（Tesseract 语言代码，多语言用 + 连接）
OCR_LANGUAGE = "chi_sim+eng"

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
    """PDF 文档加载器，支持纯文本 PDF 和图片型（扫描件）PDF"""

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
        ocr_enabled: bool = True,
        ocr_language: str = OCR_LANGUAGE,
        ocr_text_threshold: int = OCR_TEXT_THRESHOLD,
        ocr_dpi: int = 150,
        ocr_workers: int = 0,
    ):
        """
        Args:
            chunk_size: 每个文本块的最大字符数
            chunk_overlap: 块间重叠字符数
            extra_noise_patterns: 额外的噪音正则表达式列表
            ocr_enabled: 是否启用 OCR（对图片型页面）
            ocr_language: Tesseract 语言代码，如 "chi_sim+eng"
            ocr_text_threshold: 页面文字数低于此值时触发 OCR
            ocr_dpi: OCR 渲染分辨率，越高精度越好但越慢（建议 150-300）
            ocr_workers: 并行 OCR 线程数，0 表示自动 (min(8, CPU_COUNT))
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language
        self.ocr_text_threshold = ocr_text_threshold
        self.ocr_dpi = ocr_dpi
        
        # 默认不建议太多，因为 Tesseract 是 CPU 密集型
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        if ocr_workers <= 0:
            self.ocr_workers = min(8, cpu_count)
        else:
            self.ocr_workers = ocr_workers

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

    def _process_page(self, doc_path: str, page_num: int) -> Dict[str, Any]:
        """
        处理单页的任务单元（供线程池调用）
        注意：每个线程都必须重新打开文档，防止 fitz 句柄冲突
        """
        try:
            with fitz.open(doc_path) as doc:
                page = doc[page_num]
                
                # 1. 尝试常规提取
                raw_text = page.get_text("text")
                is_image_page = self.ocr_enabled and len(raw_text.strip()) < self.ocr_text_threshold

                if is_image_page:
                    # 切换 OCR
                    tp = page.get_textpage_ocr(
                        language=self.ocr_language,
                        dpi=self.ocr_dpi,
                        full=True,
                    )
                    text = page.get_text("text", textpage=tp)
                else:
                    text = raw_text

                if text.strip():
                    cleaned = self.clean_page_text(text, self.noise_patterns)
                    if cleaned:
                        return {
                            "text": cleaned,
                            "page": page_num + 1,
                            "ocr": is_image_page,
                            "error": None
                        }
            return None
        except Exception as e:
            logger.error("第 %d 页 OCR 失败: %s", page_num + 1, e)
            return {"page": page_num+1, "error": str(e)}

    def load_pdf(self, file_path: str) -> List[TextChunk]:
        """
        加载 PDF 文件并分块，支持多线程 OCR

        Args:
            file_path: PDF 文件路径

        Returns:
            TextChunk 列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF 文件不存在: {file_path}")

        logger.info("正在加载 PDF: %s（多线程 OCR=%s, 并发线程=%d）",
                    file_path, self.ocr_enabled, self.ocr_workers)
        
        # 为了获取总页数
        with fitz.open(file_path) as doc:
            total_pages = len(doc)
        
        # 1. 并行处理每页
        from concurrent.futures import ProcessPoolExecutor, as_completed
        pages_results = [None] * total_pages
        
        logger.info("开始多进程解析页面 (共 %d 页)...", total_pages)
        with ProcessPoolExecutor(max_workers=self.ocr_workers) as executor:
            futures = {
                executor.submit(self._process_page, file_path, page_num): page_num 
                for page_num in range(total_pages)
            }
            
            done = 0
            ocr_count = 0
            for future in as_completed(futures):
                page_idx = futures[future]
                done += 1
                result = future.result()
                if result and not result.get("error"):
                    pages_results[page_idx] = result
                    if result.get("ocr"):
                        ocr_count += 1
                
                # 进度打印
                if done % 20 == 0 or done == total_pages:
                    logger.info("页面处理进度: %d/%d (已识别 OCR 页: %d)", done, total_pages, ocr_count)

        # 过滤掉 None 结果
        pages = [p for p in pages_results if p is not None]
        filename = os.path.basename(file_path)

        logger.info("PDF 加载完成。有效页 %d/%d (OCR页: %d)", len(pages), total_pages, ocr_count)

        # 2. 检测章节结构 (必须保持页码顺序，前面 pages_results[page_idx] 已保证顺序)
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
            ocr = page_info.get("ocr", False)

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
                    if current_chunk:
                        chunks.append(TextChunk(
                            text=current_chunk,
                            metadata={
                                "source": filename,
                                "page": page,
                                "chapter": chapter,
                                "section": section,
                                "type": "textbook",
                                "ocr": ocr,
                            }
                        ))
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
                        "ocr": ocr,
                    }
                ))

        return chunks
