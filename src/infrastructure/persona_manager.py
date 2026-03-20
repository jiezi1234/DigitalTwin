"""
分身管理器 — 统一管理 {CHROMA_PERSIST_DIR}/personas.json
"""

import json
import os
import uuid
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class PersonaManager:
    """管理数字分身的元数据和配置"""

    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self._path = os.path.join(persist_dir, "personas.json")
        os.makedirs(persist_dir, exist_ok=True)
        logger.info(f"初始化 PersonaManager, 配置文件路径: {self._path}")

    def _load(self) -> List[dict]:
        """从本地 JSON 文件加载分身列表"""
        if not os.path.exists(self._path):
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 过滤掉 schema 定义项
            return [p for p in data if isinstance(p, dict) and p.get("_type") != "schema"]
        except Exception as e:
            logger.error(f"加载 personas.json 失败: {e}")
            return []

    def _save(self, personas: List[dict]):
        """保存分身列表到本地 JSON 文件"""
        # 读取原始数据，保留 _type=schema 的文档头条目（如果存在）
        existing = []
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                pass
        
        schema_entries = [p for p in existing if isinstance(p, dict) and p.get("_type") == "schema"]
        
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(schema_entries + personas, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存 personas.json 失败: {e}")
            raise

    def list(self) -> List[dict]:
        """列出所有分身"""
        return self._load()

    def get(self, persona_id: str) -> Optional[dict]:
        """根据 ID 获取分身信息"""
        for p in self._load():
            if p.get("id") == persona_id:
                return p
        return None

    def create(self, name: str, system_prompt: str, collection: str = None, source_type: str = "chat") -> dict:
        """创建新分身

        Args:
            source_type: 数据来源类型 ("chat" 聊天记录 / "knowledge" 知识库)
        """
        pid = str(uuid.uuid4())
        persona = {
            "id": pid,
            "name": name,
            "collection": collection or f"persona_{pid[:8]}",
            "system_prompt": system_prompt,
            "created_at": datetime.now().isoformat(),
            "doc_count": 0,
            "model_params": {},
            "source_type": source_type
        }
        personas = self._load()
        personas.append(persona)
        self._save(personas)
        logger.info(f"成功创建分身: {name} (ID: {pid})")
        return persona

    def update_doc_count(self, persona_id: str, count: int):
        """更新分身的文档数量统计"""
        personas = self._load()
        found = False
        for p in personas:
            if p.get("id") == persona_id:
                p["doc_count"] = count
                found = True
                break
        if found:
            self._save(personas)
            logger.debug(f"更新分身 {persona_id} 文档数为 {count}")

    def update_model_params(self, persona_id: str, params: dict):
        """合并更新分身的模型参数 (如 max_tokens)"""
        personas = self._load()
        found = False
        for p in personas:
            if p.get("id") == persona_id:
                p.setdefault("model_params", {}).update(params)
                found = True
                break
        if found:
            self._save(personas)
            logger.info(f"更新分身 {persona_id} 模型参数: {params}")

    def delete(self, persona_id: str) -> bool:
        """仅删除 metadata 记录，不删除向量库数据"""
        personas = self._load()
        new_list = [p for p in personas if p.get("id") != persona_id]
        if len(new_list) == len(personas):
            return False
        self._save(new_list)
        logger.info(f"已删除分身配置: {persona_id}")
        return True
