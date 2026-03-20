"""通用的文档数据模型"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import hashlib


@dataclass
class Document:
    """标准化的文档模型"""

    content: str
    """文档内容"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """元数据，例如 {"source": "chat", "chat_time": 123456, "talker": "张三"}"""

    doc_id: Optional[str] = None
    """文档唯一ID（可选，若不提供则自动生成）"""

    def __post_init__(self):
        """初始化后处理"""
        if not self.doc_id:
            # 若不提供 ID，基于内容生成简单的哈希
            self.doc_id = hashlib.md5(
                (self.content + str(self.metadata)).encode()
            ).hexdigest()
