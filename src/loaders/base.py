"""
数据加载器基类和工厂
"""

import os
import glob
import json
import hashlib
import logging
import concurrent.futures
import multiprocessing
from abc import ABC, abstractmethod
from typing import Dict, Type, List, Any, Optional, Set, Callable
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

    @staticmethod
    def generate_hash(content: str, metadata: Dict[str, Any]) -> str:
        """生成文档的唯一摘要哈希"""
        # 排除可能变动的元数据字段，如 source_file 或 timestamp
        # 重点关注内容和核心元数据
        core_meta = {
            k: v for k, v in metadata.items() 
            if k not in ("source", "source_file", "import_time", "chat_time_str")
        }
        unique_str = f"{content}|{json.dumps(core_meta, sort_keys=True)}"
        return hashlib.sha256(unique_str.encode('utf-8')).hexdigest()

    @staticmethod
    def _run_parallel(
        func: Callable, 
        tasks: List[Any], 
        max_workers: Optional[int] = None
    ) -> List[Any]:
        """
        通用的多进程并行执行包装器
        
        Args:
            func: 处理单个任务的函数（需可序列化，建议为静态方法或全局函数）
            tasks: 任务参数列表
            max_workers: 并行进程数
            
        Returns:
            有序的结果列表
        """
        if not tasks:
            return []
            
        num_workers = max_workers or min(8, multiprocessing.cpu_count())
        results = [None] * len(tasks)
        
        # 仅在需要并行时启动进程池
        if num_workers <= 1 or len(tasks) <= 1:
            return [func(task) for task in tasks]

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(func, task): i 
                for i, task in enumerate(tasks)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"并行任务执行失败 (索引 {idx}): {e}")
                    results[idx] = None
        return results

    @classmethod
    def load_batch(
        cls, 
        pattern: str, 
        incremental: bool = False, 
        tracking_file: Optional[str] = None,
        **kwargs
    ) -> List[Document]:
        """
        批量加载符合匹配模式的文件
        
        Args:
            pattern: 文件匹配模式 (glob)
            incremental: 是否启用增量模式
            tracking_file: 增量导入跟踪文件路径
            **kwargs: 传递给加载器构造函数的参数
            
        Returns:
            加载并过滤后的文档列表
        """
        files = sorted(glob.glob(pattern))
        if not files:
            logger.warning(f"未找到匹配文件: {pattern}")
            return []

        logger.info(f"找到 {len(files)} 个文件匹配模式: {pattern}")

        # 加载跟踪数据
        tracking_data = {"imported_hashes": []}
        imported_hashes: Set[str] = set()
        
        if incremental and tracking_file and os.path.exists(tracking_file):
            try:
                with open(tracking_file, 'r', encoding='utf-8') as f:
                    tracking_data = json.load(f)
                    imported_hashes = set(tracking_data.get("imported_hashes", []))
                logger.info(f"已加载增量追踪数据，包含 {len(imported_hashes)} 条记录")
            except Exception as e:
                logger.warning(f"读取追踪文件失败: {e}")

        all_docs = []
        new_hashes_count = 0

        for filepath in files:
            try:
                # 动态创建实例。假设构造函数接受 filepath
                loader = cls(filepath=filepath, **kwargs)
                docs = loader.load()
                
                if incremental:
                    filtered_docs = []
                    for doc in docs:
                        h = cls.generate_hash(doc.content, doc.metadata)
                        if h not in imported_hashes:
                            imported_hashes.add(h)
                            filtered_docs.append(doc)
                            new_hashes_count += 1
                    all_docs.extend(filtered_docs)
                else:
                    all_docs.extend(docs)
                    
            except Exception as e:
                logger.error(f"加载文件出错 {filepath}: {e}")

        # 保存更新后的跟踪数据
        if incremental and tracking_file:
            try:
                os.makedirs(os.path.dirname(tracking_file) or ".", exist_ok=True)
                tracking_data["imported_hashes"] = list(imported_hashes)
                with open(tracking_file, 'w', encoding='utf-8') as f:
                    json.dump(tracking_data, f, ensure_ascii=False, indent=2)
                logger.info(f"已保存增量追踪数据，新增 {new_hashes_count} 条记录")
            except Exception as e:
                logger.error(f"保存追踪文件失败: {e}")

        return all_docs


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
