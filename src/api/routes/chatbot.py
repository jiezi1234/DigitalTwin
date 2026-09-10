"""
聊天机器人核心路由
"""

import logging
from flask import Blueprint, request, jsonify
from src.api.config import Config
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.db_client import DBClient
from src.infrastructure.persona_manager import PersonaManager
from src.services.rag_service import RAGService
from src.rag.react_router import ReActRetrievalRouter

logger = logging.getLogger(__name__)
chat_bp = Blueprint("chat", __name__)

persona_manager = PersonaManager(Config.CHROMA_PERSIST_DIR)
db_client = None
llm_client = None

# 会话存储 (简易内存版)
sessions = {}

# RAG 服务缓存 (persona_id -> RAGService)
rag_services = {}


def get_db_client():
    global db_client
    if db_client is None:
        db_client = DBClient()
    return db_client


def get_llm_client():
    global llm_client
    if llm_client is None:
        llm_client = LLMClient()
    return llm_client


def get_rag_service(persona):
    pid = persona["id"]
    if pid not in rag_services:
        active_llm_client = get_llm_client()
        react_router = None
        if Config.REACT_ENABLED:
            react_router = ReActRetrievalRouter(
                llm_client=active_llm_client,
                model=Config.REACT_ROUTER_MODEL,
                history_messages=Config.REACT_HISTORY_MESSAGES,
            )
        rag_services[pid] = RAGService(
            llm_client=active_llm_client,
            db_client=get_db_client(),
            collection_name=persona["collection"],
            enable_coreference_resolution=Config.RAG_COREFERENCE_RESOLUTION_ENABLED,
            enable_query_rewriting=Config.RAG_QUERY_REWRITING_ENABLED,
            query_history_messages=Config.RAG_QUERY_HISTORY_MESSAGES,
            enable_neighbor_expansion=Config.RAG_NEIGHBOR_EXPANSION_ENABLED,
            neighbor_window=Config.RAG_NEIGHBOR_WINDOW,
            neighbor_anchors=Config.RAG_NEIGHBOR_ANCHORS,
            neighbor_max_results=Config.RAG_NEIGHBOR_MAX_RESULTS,
            enable_hybrid_search=Config.RAG_HYBRID_SEARCH_ENABLED,
            hybrid_candidates=Config.RAG_HYBRID_CANDIDATES,
            hybrid_rrf_k=Config.RAG_HYBRID_RRF_K,
            dense_weight=Config.RAG_DENSE_WEIGHT,
            bm25_weight=Config.RAG_BM25_WEIGHT,
            enable_reranking=Config.RAG_RERANK_ENABLED,
            rerank_model=Config.RAG_RERANK_MODEL,
            rerank_candidates=Config.RAG_RERANK_CANDIDATES,
            enable_metadata_filtering=Config.RAG_METADATA_FILTERING_ENABLED,
            timezone_offset=Config.RAG_TIMEZONE_OFFSET,
            context_record_max_chars=Config.RAG_CONTEXT_RECORD_MAX_CHARS,
            context_dedup_threshold=Config.RAG_CONTEXT_DEDUP_THRESHOLD,
            react_router=react_router,
            retrieval_enabled=Config.RAG_ENABLED,
            max_results=Config.RAG_MAX_RESULTS,
            max_context_length=Config.RAG_MAX_CONTEXT_LENGTH,
            include_metadata=Config.RAG_INCLUDE_METADATA,
        )
    return rag_services[pid]


@chat_bp.route("/chat", methods=["POST"])
def chat():
    """处理聊天请求"""
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        session_id = data.get("session_id", "default")
        persona_id = data.get("persona_id")

        if not user_message:
            return jsonify({"status": "error", "error": "消息不能为空"}), 400

        # 获取分身信息
        persona = None
        if persona_id:
            persona = persona_manager.get(persona_id)
        if not persona:
            all_personas = persona_manager.list()
            persona = all_personas[0] if all_personas else None

        if not persona:
            return (
                jsonify(
                    {"status": "error", "error": "未找到任何分身，请先通过导入脚本创建"}
                ),
                404,
            )

        # 获取会话历史
        if session_id not in sessions:
            sessions[session_id] = []
        messages = sessions[session_id]

        # 构造用户消息
        messages.append({"role": "user", "content": user_message})

        # 获取 RAG 服务并执行对话管道
        service = get_rag_service(persona)

        reply, retrieval_stats = service.chat(
            query=user_message,
            conversation=messages[:-1],  # 传入历史
            persona=persona,
            system_prefix=Config.RAG_SYSTEM_PREFIX,
            role_instruction=Config.RAG_ROLE_INSTRUCTION,
            max_tokens=persona.get("model_params", {}).get("max_tokens", 500),
        )

        # 添加助手回复到历史
        messages.append({"role": "assistant", "content": reply})

        # 限制历史长度
        if len(messages) > 40:
            sessions[session_id] = messages[-40:]

        return jsonify(
            {
                "status": "success",
                "reply": reply,
                "session_id": session_id,
                "debug": retrieval_stats,  # 返回基础检索信息供前端展示/调试
            }
        )

    except Exception as e:
        logger.error(f"聊天接口异常: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@chat_bp.route("/reset", methods=["POST"])
def reset():
    """重置会话"""
    data = request.get_json() or {}
    session_id = data.get("session_id", "default")
    if session_id in sessions:
        del sessions[session_id]
    return jsonify({"status": "success", "message": "会话已重置"})
