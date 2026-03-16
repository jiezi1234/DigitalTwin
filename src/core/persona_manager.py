"""
分身管理器 — 统一管理 {CHROMA_PERSIST_DIR}/personas.json
"""

import json
import os
import uuid
from datetime import datetime
from typing import List, Optional


class PersonaManager:
    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self._path = os.path.join(persist_dir, "personas.json")
        os.makedirs(persist_dir, exist_ok=True)

    def _load(self) -> List[dict]:
        if not os.path.exists(self._path):
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [p for p in data if p.get("_type") != "schema"]
        except Exception:
            return []

    def _save(self, personas: List[dict]):
        # 读取原始数据，保留 _type=schema 的文档头条目
        existing = []
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                pass
        schema_entries = [p for p in existing if p.get("_type") == "schema"]
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(schema_entries + personas, f, ensure_ascii=False, indent=2)

    def list(self) -> List[dict]:
        return self._load()

    def get(self, persona_id: str) -> Optional[dict]:
        for p in self._load():
            if p["id"] == persona_id:
                return p
        return None

    def create(self, name: str, system_prompt: str, collection: str = None) -> dict:
        pid = str(uuid.uuid4())
        persona = {
            "id": pid,
            "name": name,
            "collection": collection or f"persona_{pid[:8]}",
            "system_prompt": system_prompt,
            "created_at": datetime.now().isoformat(),
            "doc_count": 0,
        }
        personas = self._load()
        personas.append(persona)
        self._save(personas)
        return persona

    def update_doc_count(self, persona_id: str, count: int):
        personas = self._load()
        for p in personas:
            if p["id"] == persona_id:
                p["doc_count"] = count
                break
        self._save(personas)

    def update_model_params(self, persona_id: str, params: dict):
        """合并更新分身的 model_params 字段"""
        personas = self._load()
        for p in personas:
            if p["id"] == persona_id:
                p.setdefault("model_params", {}).update(params)
                break
        self._save(personas)

    def delete(self, persona_id: str) -> bool:
        """仅删除 personas.json 记录，不动 ChromaDB 数据"""
        personas = self._load()
        new_list = [p for p in personas if p["id"] != persona_id]
        if len(new_list) == len(personas):
            return False
        self._save(new_list)
        return True
