"""
分身管理路由
"""

from flask import Blueprint, jsonify, request
from src.api.config import Config
from src.infrastructure.persona_manager import PersonaManager

persona_bp = Blueprint("persona", __name__)
persona_manager = PersonaManager(Config.CHROMA_PERSIST_DIR)

@persona_bp.route("/api/personas", methods=["GET"])
def list_personas():
    """列出所有分身"""
    try:
        personas = persona_manager.list()
        return jsonify({"status": "success", "personas": personas})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@persona_bp.route("/api/personas/<persona_id>", methods=["DELETE"])
def delete_persona(persona_id):
    """删除分身"""
    try:
        deleted = persona_manager.delete(persona_id)
        if not deleted:
            return jsonify({"status": "error", "error": "分身不存在"}), 404
        return jsonify({"status": "success", "message": "分身已删除"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500
