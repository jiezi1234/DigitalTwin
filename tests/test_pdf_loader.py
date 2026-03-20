import os
import pytest
from src.loaders.pdf_loader import PDFLoader
from src.loaders.base import DataLoaderFactory

# 测试用 PDF 路径（使用项目中现有的测试文件）
TEST_PDF = "data/pdf/test.pdf"

def test_pdf_loader_initialization():
    """测试 PDF 加载器初始化参数"""
    loader = PDFLoader(
        filepath=TEST_PDF,
        ocr_enabled=False,
        chunk_size=500,
        chunk_overlap=50
    )
    assert loader.filepath == TEST_PDF
    assert loader.ocr_enabled is False
    assert loader.chunk_size == 500
    assert loader.chunk_overlap == 50

def test_pdf_loader_load_real_file():
    """测试实际加载 PDF 文件（如果文件存在）"""
    if not os.path.exists(TEST_PDF):
        pytest.skip(f"测试文件不存在: {TEST_PDF}")

    loader = PDFLoader(filepath=TEST_PDF, ocr_enabled=True)
    docs = loader.load()
    
    assert len(docs) > 0
    # 检查标准元数据
    doc = docs[0]
    assert doc.metadata["source"] == "pdf"
    assert "page" in doc.metadata
    assert "ocr" in doc.metadata
    assert "content_type" in doc.metadata
    assert doc.metadata["content_type"] == "textbook"

def test_pdf_loader_factory_registration():
    """工厂应该能正确创建重构后的 PDF 加载器"""
    DataLoaderFactory.register("pdf", PDFLoader)
    loader = DataLoaderFactory.create("pdf", filepath=TEST_PDF, chunk_size=800)
    
    assert isinstance(loader, PDFLoader)
    assert loader.chunk_size == 800

