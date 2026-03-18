"""
数据加载器基类和工厂
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Type, List, Any
from src.infrastructure.document import Document

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """数据加载器基类"""

    @abstractmethod
    def load(self) -> List[Document]:
        """
        加载数据并返回标准化的 Document 列表

        Returns:
            Document 列表
        """
        pass


class DataLoaderFactory:
    """数据加载器工厂"""

    _loaders: Dict[str, Type[DataLoader]] = {}

    @classmethod
    def register(cls, loader_type: str, loader_class: Type[DataLoader]):
        """
        注册加载器类

        Args:
            loader_type: 加载器类型标识（如 "csv"、"pdf"）
            loader_class: 加载器类
        """
        cls._loaders[loader_type] = loader_class
        logger.info(f"注册加载器: {loader_type} -> {loader_class.__name__}")

    @classmethod
    def create(cls, loader_type: str, **kwargs) -> DataLoader:
        """
        创建指定类型的加载器

        Args:
            loader_type: 加载器类型标识
            **kwargs: 传给加载器的参数

        Returns:
            DataLoader 实例

        Raises:
            ValueError: 如果加载器类型不存在
        """
        if loader_type not in cls._loaders:
            raise ValueError(
                f"未知的加载器类型: {loader_type}。"
                f"已注册的类型: {', '.join(cls._loaders.keys())}"
            )

        loader_class = cls._loaders[loader_type]
        logger.info(f"创建加载器: {loader_type}")
        return loader_class(**kwargs)

    @classmethod
    def list_loaders(cls) -> List[str]:
        """列出所有已注册的加载器"""
        return list(cls._loaders.keys())
