import pytest
from unittest.mock import patch, MagicMock
from src.loaders.pdf_loader import PDFLoader
from src.loaders.base import DataLoaderFactory


def test_pdf_loader_initialization():
    """初始化 PDF 加载器"""
    loader = PDFLoader(filepath="test.pdf")
    assert loader.filepath == "test.pdf"


def test_pdf_loader_attributes():
    """测试 PDF 加载器属性"""
    loader = PDFLoader(filepath="test.pdf", extract_metadata=False)
    assert loader.filepath == "test.pdf"
    assert loader.extract_metadata is False


def test_pdf_loader_factory_registration():
    """工厂应该能创建 PDF 加载器"""
    DataLoaderFactory.register("pdf", PDFLoader)

    loader = DataLoaderFactory.create("pdf", filepath="test.pdf")
    assert isinstance(loader, PDFLoader)

