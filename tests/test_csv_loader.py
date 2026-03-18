import pytest
import tempfile
import csv
from src.loaders.csv_loader import CSVLoader
from src.loaders.base import DataLoaderFactory


@pytest.fixture
def sample_csv_file():
    """创建示例 CSV 文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['talker', 'message', 'chat_time'])
        writer.writerow(['张三', '你好', '1609459200'])
        writer.writerow(['李四', '你好啊', '1609459260'])
        f.flush()
        yield f.name


def test_csv_loader_initialization(sample_csv_file):
    """初始化 CSV 加载器"""
    loader = CSVLoader(filepath=sample_csv_file)
    assert loader.filepath == sample_csv_file


def test_csv_loader_load(sample_csv_file):
    """加载 CSV 文件"""
    loader = CSVLoader(filepath=sample_csv_file)
    documents = loader.load()

    assert len(documents) == 2
    assert documents[0].content == "你好"
    assert documents[0].metadata['talker'] == "张三"


def test_csv_loader_factory_registration(sample_csv_file):
    """工厂应该能创建 CSV 加载器"""
    DataLoaderFactory.register("csv", CSVLoader)

    loader = DataLoaderFactory.create("csv", filepath=sample_csv_file)
    documents = loader.load()

    assert len(documents) == 2
