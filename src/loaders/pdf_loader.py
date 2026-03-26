import os
import re
import json
import hashlib
import fitz  # PyMuPDF
import logging
from datetime import datetime, timezone
import concurrent.futures
import multiprocessing
from typing import List, Dict, Any, Optional
from src.loaders.base import DataLoader
from src.infrastructure.document import Document
from src.infrastructure.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

# 默认噪音模式 (同步自旧项目)
DEFAULT_NOISE_PATTERNS = [
    re.compile(r'^Principle and Technology of Database\s*$', re.MULTILINE),
    re.compile(r'^NOTES\s*$', re.MULTILINE),
    re.compile(r'^Copyright\s*©.*$', re.MULTILINE),
    re.compile(r'^Page\s+\d+\s*$', re.MULTILINE),
]

# 章节标题提取正则
CHAPTER_PATTERNS = [
    re.compile(r'^第[一二三四五六七八九十\d]+章\s+'),
    re.compile(r'^第[一二三四五六七八九十\d]+节\s+'),
    re.compile(r'^\d+\.\d+(\.\d+)?\s+'),  # 1.1 / 1.1.1 格式
    re.compile(r'^Chapter\s+\d+', re.IGNORECASE),
]


class PDFLoader(DataLoader):
    """高级 PDF 加载器，支持纯文本和扫描件 OCR (多进程并行加速)"""

    def __init__(
        self,
        filepath: str,
        ocr_enabled: bool = True,
        ocr_language: str = "chi_sim+eng",
        ocr_text_threshold: int = 50,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_workers: Optional[int] = None,
        **kwargs
    ):
        """
        初始化 PDF 加载器

        Args:
            filepath: PDF 文件路径
            ocr_enabled: 是否对图片页启用 OCR
            ocr_language: Tesseract 语言配置 (如 "chi_sim+eng")
            ocr_text_threshold: 页面文字字符数低于此值则视为图片页面，启用 OCR
            chunk_size: 分块大小 (字符数)
            chunk_overlap: 块间重叠度
            max_workers: 并行进程数 (缺省为 min(8, cpu_count))
        """
        self.filepath = filepath
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language
        self.ocr_text_threshold = ocr_text_threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if max_workers is None:
            self.max_workers = min(8, multiprocessing.cpu_count())
        else:
            self.max_workers = max_workers

    def _clean_text(self, text: str) -> str:
        """清理页面文本噪音"""
        for pattern in DEFAULT_NOISE_PATTERNS:
            text = pattern.sub('', text)
        # 清理多余空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def _normalize_bbox(bbox: Any) -> Dict[str, float]:
        """将 PyMuPDF 的 bbox 统一为 JSON 友好的坐标格式"""
        x0, y0, x1, y1 = bbox
        return {
            "x0": round(float(x0), 3),
            "y0": round(float(y0), 3),
            "x1": round(float(x1), 3),
            "y1": round(float(y1), 3),
        }

    def export_structured(
        self,
        output_dir: str,
        json_filename: str = "structured_content.json",
        image_dirname: str = "images",
    ) -> Dict[str, Any]:
        """
        导出 PDF 的文本块、图片及统一 JSON 清单。

        Args:
            output_dir: 导出目录
            json_filename: 统一 JSON 文件名
            image_dirname: 图片输出子目录名

        Returns:
            结构化导出结果字典
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"PDF 文件不存在: {self.filepath}")

        os.makedirs(output_dir, exist_ok=True)
        images_dir = os.path.join(output_dir, image_dirname)
        os.makedirs(images_dir, exist_ok=True)

        filename = os.path.basename(self.filepath)
        export_data: Dict[str, Any] = {
            "source": "pdf",
            "source_file": filename,
            "source_path": os.path.abspath(self.filepath),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pages": [],
            "summary": {
                "page_count": 0,
                "text_block_count": 0,
                "image_count": 0,
            },
        }

        with fitz.open(self.filepath) as doc:
            export_data["summary"]["page_count"] = len(doc)
            written_images: Dict[str, str] = {}

            for page_index, page in enumerate(doc):
                page_dict = page.get_text("dict")
                page_entry = {
                    "page": page_index + 1,
                    "width": round(float(page.rect.width), 3),
                    "height": round(float(page.rect.height), 3),
                    "text_blocks": [],
                    "images": [],
                }

                image_index = 0
                for block in page_dict.get("blocks", []):
                    block_type = block.get("type")

                    if block_type == 0:
                        spans = []
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                spans.append(span.get("text", ""))

                        raw_text = "".join(spans)
                        cleaned_text = self._clean_text(raw_text)
                        if not cleaned_text:
                            continue

                        page_entry["text_blocks"].append({
                            "block_index": len(page_entry["text_blocks"]),
                            "bbox": self._normalize_bbox(block["bbox"]),
                            "content": cleaned_text,
                        })
                        export_data["summary"]["text_block_count"] += 1

                    elif block_type == 1:
                        image_bytes = block.get("image")
                        image_ext = (block.get("ext") or "bin").lower()
                        image_index += 1
                        image_hash = hashlib.sha256(image_bytes or b"").hexdigest()
                        canonical_relative_path = written_images.get(image_hash)

                        if canonical_relative_path is None:
                            image_filename = (
                                f"page_{page_index + 1:04d}_img_{image_index:03d}.{image_ext}"
                            )
                            canonical_relative_path = os.path.join(image_dirname, image_filename)
                            image_path = os.path.join(output_dir, canonical_relative_path)

                            if image_bytes:
                                with open(image_path, "wb") as image_file:
                                    image_file.write(image_bytes)
                            written_images[image_hash] = canonical_relative_path

                        page_entry["images"].append({
                            "image_index": image_index - 1,
                            "bbox": self._normalize_bbox(block["bbox"]),
                            "width": block.get("width"),
                            "height": block.get("height"),
                            "colorspace": block.get("colorspace"),
                            "bpc": block.get("bpc"),
                            "xres": block.get("xres"),
                            "yres": block.get("yres"),
                            "extension": image_ext,
                            "file": canonical_relative_path.replace(os.sep, "/"),
                            "image_hash": image_hash,
                            "size_bytes": len(image_bytes or b""),
                            "is_duplicate": image_hash in written_images and written_images[image_hash] != canonical_relative_path,
                        })
                        export_data["summary"]["image_count"] += 1

                export_data["pages"].append(page_entry)

        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(export_data, json_file, ensure_ascii=False, indent=2)

        export_data["json_file"] = os.path.abspath(json_path)
        export_data["images_dir"] = os.path.abspath(images_dir)
        return export_data

    def load(self) -> List[Document]:
        """执行 PDF 加载、通过基类并行执行 OCR 及分块"""
        with tracer.start_as_current_span("pdf_loader.load") as span:
            span.set_attribute("loader.filepath", self.filepath)

            if not os.path.exists(self.filepath):
                raise FileNotFoundError(f"PDF 文件不存在: {self.filepath}")

            filename = os.path.basename(self.filepath)
            
            # 1. 获取总页数
            with fitz.open(self.filepath) as doc:
                total_pages = len(doc)
                
                # 提前测试 Tesseract 依赖，避免多进程重复报错刷屏
                if self.ocr_enabled and total_pages > 0:
                    try:
                        # 用第一页轻轻地初始化一下 OCR
                        doc[0].get_textpage_ocr(language=self.ocr_language, dpi=72, full=False)
                    except Exception as e:
                        if "Tesseract is not installed" in str(e) or "tessdata" in str(e):
                            logger.warning(f"由于未安装 Tesseract OCR 或未配置 tessdata，已自动回退为纯文字提取模式，跳过所有 OCR 处理。({e})")
                            self.ocr_enabled = False
                        else:
                            logger.warning(f"OCR 初始化异常: {e}")
            
            # 2. 准备并行任务参数
            tasks = [
                {
                    "filepath": self.filepath,
                    "page_num": i,
                    "ocr_enabled": self.ocr_enabled,
                    "ocr_language": self.ocr_language,
                    "ocr_text_threshold": self.ocr_text_threshold
                }
                for i in range(total_pages)
            ]
            
            # 3. 调用基类通用并行方法
            logger.info(f"调用基类并行组件解析 PDF (共 {total_pages} 页, 并发: {self.max_workers})...")
            pages_data = self._run_parallel(
                PDFLoader._process_single_page_static, 
                tasks, 
                max_workers=self.max_workers
            )

            # 4. 顺序处理结果，进行章节检测和分块
            documents = []
            current_chapter = "前言"
            current_section = ""

            for p_data in pages_data:
                if not p_data or p_data.get("error") or not p_data.get("text"):
                    if p_data and p_data.get("error"):
                        logger.error(f"处理第 {p_data['page_num']+1} 页失败: {p_data['error']}")
                    continue

                text = p_data["text"]
                page_idx = p_data["page_num"]
                is_ocr = p_data["is_ocr"]

                # 章节检测 (必须按顺序进行)
                current_chapter, current_section = self._detect_headings(text, current_chapter, current_section)

                # 分块
                page_chunks = self._split_text(text)
                for i, chunk_text in enumerate(page_chunks):
                    metadata = {
                        "source": "pdf",
                        "source_file": filename,
                        "page": page_idx + 1,
                        "ocr": is_ocr,
                        "chapter": current_chapter,
                        "section": current_section,
                        "chunk_index": i,
                        "content_type": "textbook"
                    }
                    documents.append(Document(
                        content=chunk_text,
                        metadata=metadata
                    ))

            logger.info(f"PDF 加载完成: {filename}, 共生成 {len(documents)} 个 Doc 块")
            span.set_attribute("loader.docs_count", len(documents))
            return documents

    @staticmethod
    def _process_single_page_static(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        静态单页处理逻辑，避免实例序列化问题
        """
        filepath = params["filepath"]
        page_num = params["page_num"]
        ocr_enabled = params["ocr_enabled"]
        ocr_language = params["ocr_language"]
        ocr_text_threshold = params["ocr_text_threshold"]

        try:
            with fitz.open(filepath) as doc:
                page = doc[page_num]
                raw_text = page.get_text("text")
                is_ocr = False

                if ocr_enabled and len(raw_text.strip()) < ocr_text_threshold:
                    try:
                        tp = page.get_textpage_ocr(
                            language=ocr_language,
                            dpi=150,
                            full=True
                        )
                        raw_text = page.get_text("text", textpage=tp)
                        is_ocr = True
                    except Exception as ocr_err:
                        logger.warning(f"页面 {page_num+1} OCR 失败: {ocr_err}")

                # 注意：_clean_text 现在逻辑也在静态方法里或需重复定义
                # 为简单起见，这里直接复用逻辑
                text = raw_text
                # 默认噪音模式 (同步自旧项目)
                noise_patterns = [
                    re.compile(r'^Principle and Technology of Database\s*$', re.MULTILINE),
                    re.compile(r'^NOTES\s*$', re.MULTILINE),
                    re.compile(r'^Copyright\s*©.*$', re.MULTILINE),
                    re.compile(r'^Page\s+\d+\s*$', re.MULTILINE),
                ]
                for pattern in noise_patterns:
                    text = pattern.sub('', text)
                text = re.sub(r'\n{3,}', '\n\n', text)
                cleaned_text = text.strip()

                return {
                    "page_num": page_num,
                    "text": cleaned_text,
                    "is_ocr": is_ocr,
                    "error": None
                }
        except Exception as e:
            return {
                "page_num": page_num,
                "text": "",
                "is_ocr": False,
                "error": str(e)
            }

    def _detect_headings(self, text: str, current_chapter: str, current_section: str):
        """尝试从文本更新当前章节/小节信息"""
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line: continue
            for pattern in CHAPTER_PATTERNS:
                if pattern.match(line):
                    if '章' in line or line.lower().startswith('chapter'):
                        current_chapter = line
                        current_section = ""
                    else:
                        current_section = line
                    break # 找到一个匹配后跳出内层模式循环
        return current_chapter, current_section

    def _split_text(self, text: str) -> List[str]:
        """将页面文本根据 chunk_size 和 overlap 进行分块"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= len(text):
                break
                
            start = end - self.chunk_overlap
            if start >= end:
                start = end
        
        return chunks
