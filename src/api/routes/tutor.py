"""
数字助教路由 (Tutor) — Agent 架构版
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
from src.infrastructure.text_embedding_client import TextEmbeddingClient
from src.services.textbook_rag_service import TextbookRAGService
from src.infrastructure.multimodal_embedding_client import MultiModalEmbeddingClient
from src.services.multimodal_pdf_service import MultiModalPDFIndexService
from src.agent.tutor_react_agent import TutorReActAgent
from src.skills.textbook_retrieval.scripts.textbook_text_skill import TextbookTextSkill
from src.skills.textbook_retrieval.scripts.textbook_image_skill import TextbookImageSkill

logger = logging.getLogger(__name__)
tutor_bp = Blueprint("tutor", __name__)

# 初始化基础设施
db_client = DBClient()
llm_client = LLMClient()
mm_client = MultiModalEmbeddingClient(model=Config.MM_EMBED_MODEL)
text_client = TextEmbeddingClient(model=Config.EMBED_MODEL)
tutor_sessions = {}

# 懒加载 — 整个助教 Agent（含两个技能）
_tutor_agent: TutorReActAgent = None
_tutor_rag_service: TextbookRAGService = None


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


def get_tutor_rag_service() -> TextbookRAGService:
    global _tutor_rag_service
    if _tutor_rag_service is None:
        try:
            _tutor_rag_service = TextbookRAGService(
                llm_client=llm_client,
                db_client=db_client,
                text_collection_name=Config.TUTOR_MM_TEXT_COLLECTION,
                image_collection_name=Config.TUTOR_MM_IMAGE_COLLECTION,
                ocr_collection_name=Config.TUTOR_OCR_TEXT_COLLECTION,
                enable_query_rewriting=False,
                mm_client=mm_client,
                text_client=text_client,
            )
            logger.info("[Tutor] TextbookRAGService 初始化完成")
        except Exception as e:
            logger.warning(f"[Tutor] RAG Service 初始化失败: {e}")
    return _tutor_rag_service


def get_tutor_agent() -> TutorReActAgent:
    global _tutor_agent
    if _tutor_agent is None:
        rag_service = get_tutor_rag_service()
        if rag_service is None:
            return None
        try:
            text_skill = TextbookTextSkill(rag_service=rag_service, text_k=8)
            image_skill = TextbookImageSkill(rag_service=rag_service, image_k=4)
            _tutor_agent = TutorReActAgent(
                llm_client=llm_client,
                tools=[text_skill, image_skill],
                max_iterations=5,
            )
            logger.info("[Tutor] TutorReActAgent 初始化完成，已加载技能: search_textbook_text, search_textbook_images")
        except Exception as e:
            logger.warning(f"[Tutor] Agent 初始化失败: {e}")
    return _tutor_agent


@tutor_bp.route("/tutor/chat", methods=["POST"])
def tutor_chat():
    """助教对话 — ReAct Agent 架构 (支持流式)"""
    data = request.get_json()
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "tutor-default")
    stream = data.get("stream", True)

    if not user_message:
        return jsonify({"status": "error", "error": "消息不能为空"}), 400

    logger.info(f"[Tutor Agent] 新请求汇入 session={session_id}: {user_message}")

    agent = get_tutor_agent()
    rag_service = get_tutor_rag_service()

    if session_id not in tutor_sessions:
        tutor_sessions[session_id] = []
    messages = tutor_sessions[session_id]
    messages.append({"role": "user", "content": user_message})

    # ── Agent 推理（非流式，内部思考 + 工具调用）──────────────────────
    reply_text = ""
    image_hits = []
    text_results = []

    if agent:
        try:
            reply_text, image_hits, text_results = agent.run(
                query=user_message,
                conversation_history=messages[:-1],
                max_tokens=Config.TUTOR_MAX_TOKENS,
                model=Config.TUTOR_VL_MODEL,
            )
        except Exception as e:
            logger.error(f"[Tutor Agent] Agent 推理异常: {e}")
            reply_text = "抱歉，处理你的问题时出现了错误，请稍后再试。"
    else:
        logger.warning("[Tutor Agent] Agent 未初始化，返回兜底回复。")
        reply_text = "助教服务尚未就绪（可能未导入课本），请先通过控制台喂入教材资料。"

    # ── 若有图片命中，附加图片内容给 LLM 进行多模态最终生成 ─────────────
    # Agent 已经内部调用了 qwen-vl-plus 做最终回答（含思考），
    # 此处直接推流已得到的 reply_text，保持与前端协议兼容。
    # ── 清洗 reply：去掉 LLM 可能自己写的"参考资料"尾巴，系统会单独渲染引用 ──
    import re as _re
    reply_text = _re.sub(
        r'\n*[\-—\*]*\s*(参考资料|参考来源|引用来源|参考文献|来源)[：:：]?[\s\S]*$',
        '',
        reply_text,
        flags=_re.IGNORECASE,
    ).rstrip()

    messages.append({"role": "assistant", "content": reply_text})
    # 提取 sources（基于 reply 中的 [1][2] 引用编号）
    # 如果 LLM 没有显式写 [1][2]，则自动展示所有命中段落的来源（最多5条）
    if rag_service and text_results:
        sources, mapping = rag_service.get_sources(text_results, reply=reply_text)
        if not sources:
            # Fallback: no citation markers found, include all unique sources
            sources, mapping = rag_service.get_sources(text_results, reply=None)
            
        if mapping:
            import re as _re
            def repl_cite(m):
                idx = int(m.group(1))
                if idx in mapping:
                    return f"[{mapping[idx]}]"
                return m.group(0)
            reply_text = _re.sub(r'\[(\d+)\]', repl_cite, reply_text)
            messages[-1]["content"] = reply_text
    else:
        sources = []

    # 会话截断
    if len(messages) > 40:
        tutor_sessions[session_id] = messages[-40:]

    if stream:
        def generate():
            # 字符流式推送（逐字符，前端体验）
            for char in reply_text:
                yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            yield f"data: {json.dumps({'type': 'images', 'images': image_hits})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    else:
        return jsonify({
            "status": "success",
            "reply": reply_text,
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
            global _tutor_rag_service, _tutor_agent
            _tutor_rag_service = None
            _tutor_agent = None
            logger.info("后台课本导入完成，Agent 将在下次请求时重新初始化")
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
    rag_service = get_tutor_rag_service()
    if not rag_service:
        return jsonify({"status": "error", "error": "助教服务未初始化"}), 500
    agent = get_tutor_agent()
    stats = rag_service.get_stats()
    stats["agent_ready"] = agent is not None
    stats["agent_tools"] = list(agent.tools.keys()) if agent else []
    return jsonify({"status": "success", "data": stats})
