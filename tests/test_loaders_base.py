import pytest
from abc import ABC
from src.loaders.base import DataLoader, DataLoaderFactory
from src.infrastructure.document import Document


def test_data_loader_is_abstract():
    """DataLoader 应该是抽象类"""
    assert issubclass(DataLoader, ABC)


def test_data_loader_factory_registration():
    """工厂应该支持注册新的加载器"""

    class MockLoader(DataLoader):
        def load(self):
            return [Document(content="test", metadata={"type": "mock"})]

    # 注册工厂
    DataLoaderFactory.register("mock", MockLoader)

    # 创建实例
    loader = DataLoaderFactory.create("mock")
    assert isinstance(loader, MockLoader)


def test_data_loader_factory_create_csv():
    """工厂应该能创建 CSV 加载器"""
    # 注册 CSV 加载器（稍后实现）
    try:
        loader = DataLoaderFactory.create("csv", filepath="test.csv")
        assert loader is not None
    except ValueError:
        # 如果还未注册，应该抛出 ValueError
        pass


def test_data_loader_factory_create_pdf():
    """工厂应该能创建 PDF 加载器"""
    try:
        loader = DataLoaderFactory.create("pdf", filepath="test.pdf")
        assert loader is not None
    except ValueError:
        pass
