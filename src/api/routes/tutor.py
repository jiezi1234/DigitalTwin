"""
数字助教路由 (Tutor)
"""

import logging
import json
import os
import base64
import mimetypes
import threading
from flask import Blueprint, request, jsonify, Response, stream_with_context
from src.api.config import Config
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.db_client import DBClient
from src.services.textbook_rag_service import TextbookRAGService
from src.infrastructure.multimodal_embedding_client import MultiModalEmbeddingClient
from src.services.multimodal_pdf_service import MultiModalPDFIndexService

logger = logging.getLogger(__name__)
tutor_bp = Blueprint("tutor", __name__)

# 初始化基础设施
db_client = DBClient()
llm_client = LLMClient()
mm_client = MultiModalEmbeddingClient(model=Config.MM_EMBED_MODEL)
tutor_sessions = {}

# 懒加载 TextbookRAGService
_tutor_rag_service = None


def _local_image_to_data_url(relative_path: str) -> str:
    image_path = os.path.join(Config.PDF_EXPORT_ROOT, relative_path)
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/png"
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _build_multimodal_messages(
    system_content: str,
    conversation: list,
    user_message: str,
    image_hits: list,
) -> list:
    messages = [{"role": "system", "content": system_content}]

    for item in conversation:
        messages.append(item)

    content = [{"type": "text", "text": user_message}]
    for image in image_hits[:3]:
        image_path = image.get("image_path")
        if not image_path:
            continue
        try:
            content.append({
                "type": "image_url",
                "image_url": {"url": _local_image_to_data_url(image_path)},
            })
        except Exception as exc:
            logger.warning(f"读取命中图片失败，已跳过: {image_path} - {exc}")

    messages.append({"role": "user", "content": content})
    return messages

def get_tutor_service():
    global _tutor_rag_service
    if _tutor_rag_service is None:
        try:
            _tutor_rag_service = TextbookRAGService(
                llm_client=llm_client,
                db_client=db_client,
                text_collection_name=Config.TUTOR_MM_TEXT_COLLECTION,
                image_collection_name=Config.TUTOR_MM_IMAGE_COLLECTION,
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
    image_context = ""
    results = []
    image_results = []
    image_hits = []
    if service:
        try:
            retrieval = service.retrieve(user_message, text_k=8, image_k=4)
            results = retrieval["text_results"]
            image_results = retrieval["image_results"]
            if results:
                context_text = service.format_context(
                    results, max_context_length=Config.TUTOR_MAX_CONTEXT_LENGTH
                )
            if image_results:
                image_context = service.format_image_context(image_results)
                image_hits = service.serialize_images(image_results)
            logger.info(
                "[Tutor Retrieval] session=%s text_hits=%d image_hits=%d vl_model=%s",
                session_id,
                len(results),
                len(image_hits),
                Config.TUTOR_VL_MODEL,
            )
        except Exception as e:
            logger.error(f"助教检索异常: {e}")

    # 构建 Prompt
    system_content = Config.TUTOR_SYSTEM_PROMPT
    if context_text:
        system_content = f"以下是相关课本内容：\n\n{context_text}\n\n{Config.TUTOR_SYSTEM_PROMPT}"
    if image_context:
        system_content = (
            f"{system_content}\n\n以下是检索到的相关图片及其附近文字：\n\n{image_context}\n\n"
            "如果检索结果中提供了图片标记，例如 [图1]、[图2]，说明这些图片已经可供你直接引用。"
            "当图片有助于回答时，请在合适的位置直接插入对应标记，例如 [图1]。"
            "这些标记会由系统渲染为实际图片，因此不要说“我无法展示图片”“我不能显示图片”"
            "“我只能描述图片”或类似表述。"
            "当用户明确要求展示图片、给出图片或结合图片说明时，优先至少引用一张最相关图片。"
            "不要输出图片 URL，也不要在结尾重复罗列图片；只使用 [图1]、[图2] 这类标记。"
        )

    history_messages = messages[:-1]
    gen_messages = _build_multimodal_messages(
        system_content=system_content,
        conversation=history_messages,
        user_message=user_message,
        image_hits=image_hits,
    )

    if stream:
        def generate():
            full_reply = []
            logger.info(
                "[Tutor Generate] session=%s text_hits=%d image_hits=%d vl_model=%s stream=%s",
                session_id,
                len(results),
                len(image_hits),
                Config.TUTOR_VL_MODEL,
                True,
            )

            # 调用 LLM 流式接口
            for chunk in llm_client.call_stream(
                gen_messages,
                max_tokens=Config.TUTOR_MAX_TOKENS,
                model=Config.TUTOR_VL_MODEL,
            ):
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                full_reply.append(chunk)

            # 回答完成后，根据实际引用提取 sources
            reply_text = "".join(full_reply)
            sources = service.get_sources(results, reply=reply_text) if service and results else []
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            yield f"data: {json.dumps({'type': 'images', 'images': image_hits})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            if full_reply:
                messages.append({"role": "assistant", "content": reply_text})

            # 会话截断
            if len(messages) > 40:
                tutor_sessions[session_id] = messages[-40:]

        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    else:
        logger.info(
            "[Tutor Generate] session=%s text_hits=%d image_hits=%d vl_model=%s stream=%s",
            session_id,
            len(results),
            len(image_hits),
            Config.TUTOR_VL_MODEL,
            False,
        )
        reply = llm_client.call(
            gen_messages,
            max_tokens=Config.TUTOR_MAX_TOKENS,
            model=Config.TUTOR_VL_MODEL,
        )
        messages.append({"role": "assistant", "content": reply})

        # 根据实际引用提取 sources
        sources = service.get_sources(results, reply=reply) if service and results else []

        # 会话截断
        if len(messages) > 40:
            tutor_sessions[session_id] = messages[-40:]

        return jsonify({
            "status": "success",
            "reply": reply,
            "sources": sources,
            "images": image_hits,
        })

@tutor_bp.route("/tutor/import", methods=["POST"])
def tutor_import():
    """触发课本导入 (后台)"""
    def do_import():
        try:
            index_service = MultiModalPDFIndexService(
                db_client=db_client,
                embedding_client=mm_client,
            )
            pattern = os.path.join(Config.PROJECT_ROOT, "data/pdf/*.pdf")
            index_service.index_pattern(
                pattern=pattern,
                output_root=Config.PDF_EXPORT_ROOT,
                text_collection=Config.TUTOR_MM_TEXT_COLLECTION,
                image_collection=Config.TUTOR_MM_IMAGE_COLLECTION,
            )
            global _tutor_rag_service
            _tutor_rag_service = None
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


@tutor_bp.route("/tutor/stats", methods=["GET"])
def tutor_stats():
    service = get_tutor_service()
    if not service:
        return jsonify({"status": "error", "error": "助教服务未初始化"}), 500
    return jsonify({"status": "success", "data": service.get_stats()})
