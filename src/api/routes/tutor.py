"""
数字助教路由 (Tutor)
"""

import logging
import json
import threading
from flask import Blueprint, request, jsonify, Response, stream_with_context
from src.api.config import Config
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.db_client import DBClient
from src.services.textbook_rag_service import TextbookRAGService
from src.services.import_service import ImportService
from src.loaders.pdf_loader import PDFLoader

logger = logging.getLogger(__name__)
tutor_bp = Blueprint("tutor", __name__)

# 初始化基础设施
db_client = DBClient()
llm_client = LLMClient()
tutor_sessions = {}

# 懒加载 TextbookRAGService
_tutor_rag_service = None

def get_tutor_service():
    global _tutor_rag_service
    if _tutor_rag_service is None:
        try:
            _tutor_rag_service = TextbookRAGService(
                llm_client=llm_client,
                db_client=db_client,
                collection_name=Config.TUTOR_COLLECTION
            )
        except Exception as e:
            logger.warning(f"助教服务初始化失败 (可能未导入课本): {e}")
    return _tutor_rag_service

@tutor_bp.route("/tutor/chat", methods=["POST"])
def tutor_chat():
    """助教对话 (支持流式)"""
    data = request.get_json()
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "tutor-default")
    stream = data.get("stream", True)

    if not user_message:
        return jsonify({"status": "error", "error": "消息不能为空"}), 400

    logger.debug(f"[用户输入] session={session_id} | {user_message}")

    service = get_tutor_service()
    if session_id not in tutor_sessions:
        tutor_sessions[session_id] = []
    messages = tutor_sessions[session_id]
    messages.append({"role": "user", "content": user_message})

    # RAG 检索
    context_text = ""
    results = []
    if service:
        try:
            results = service.search(user_message, k=8)
            if results:
                context_text = service.format_context(
                    results, max_context_length=Config.TUTOR_MAX_CONTEXT_LENGTH
                )
        except Exception as e:
            logger.error(f"助教检索异常: {e}")

    # 构建 Prompt
    system_content = Config.TUTOR_SYSTEM_PROMPT
    if context_text:
        system_content = f"以下是相关课本内容：\n\n{context_text}\n\n{Config.TUTOR_SYSTEM_PROMPT}"

    gen_messages = [{"role": "system", "content": system_content}]
    gen_messages.extend(messages)

    if stream:
        def generate():
            full_reply = []

            # 调用 LLM 流式接口
            for chunk in llm_client.call_stream(gen_messages, max_tokens=Config.TUTOR_MAX_TOKENS):
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                full_reply.append(chunk)

            # 回答完成后，根据实际引用提取 sources
            reply_text = "".join(full_reply)
            sources = service.get_sources(results, reply=reply_text) if service and results else []
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            if full_reply:
                messages.append({"role": "assistant", "content": reply_text})

            # 会话截断
            if len(messages) > 40:
                tutor_sessions[session_id] = messages[-40:]

        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    else:
        reply = llm_client.call(gen_messages, max_tokens=Config.TUTOR_MAX_TOKENS)
        messages.append({"role": "assistant", "content": reply})

        # 根据实际引用提取 sources
        sources = service.get_sources(results, reply=reply) if service and results else []

        # 会话截断
        if len(messages) > 40:
            tutor_sessions[session_id] = messages[-40:]

        return jsonify({
            "status": "success",
            "reply": reply,
            "sources": sources
        })

@tutor_bp.route("/tutor/import", methods=["POST"])
def tutor_import():
    """触发课本导入 (后台)"""
    def do_import():
        try:
            importer = ImportService(db_client)
            # 使用 refactored loader 和 service
            importer.import_documents(
                loader_cls=PDFLoader,
                pattern=os.path.join(Config.PROJECT_ROOT, "textbook/*.pdf"),
                collection_name=Config.TUTOR_COLLECTION,
                incremental=True
            )
            logger.info("后台课本导入完成")
        except Exception as e:
            logger.error(f"后台课本导入失败: {e}")

    threading.Thread(target=do_import, daemon=True).start()
    return jsonify({"status": "success", "message": "已在后台启动导入"})

@tutor_bp.route("/tutor/reset", methods=["POST"])
def tutor_reset():
    session_id = request.get_json().get("session_id", "tutor-default")
    tutor_sessions.pop(session_id, None)
    return jsonify({"status": "success"})
