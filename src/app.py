"""
Flask聊天机器人服务器
集成RAG检索和Qwen大模型API，支持多分身
"""

import os
import sys

# 在 chromadb/grpc 库加载前设置，抑制 C++ 层的 abseil/gRPC 噪音日志
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_TRACE"] = ""
os.environ["GLOG_minloglevel"] = "3"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"
import json
import time
import requests
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any, Tuple
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler

load_dotenv()

# ── 日志配置 ──────────────────────────────────────────────
_LOG_LEVEL = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
_LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
_LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))   # 默认 10 MB
_LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

_fmt = logging.Formatter(
    "[%(levelname)s] %(asctime)s %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_console_hdl = logging.StreamHandler()
_console_hdl.setLevel(_LOG_LEVEL)
_console_hdl.setFormatter(_fmt)

_log_dir = os.path.dirname(_LOG_FILE)
if _log_dir:
    os.makedirs(_log_dir, exist_ok=True)
_file_hdl = RotatingFileHandler(
    _LOG_FILE, maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUP_COUNT, encoding="utf-8"
)
_file_hdl.setLevel(logging.DEBUG)   # 文件始终保留完整 DEBUG 日志
_file_hdl.setFormatter(_fmt)

logging.root.setLevel(logging.DEBUG)
logging.root.addHandler(_console_hdl)
logging.root.addHandler(_file_hdl)


class _StaticFilter(logging.Filter):
    """将静态文件访问日志降级为 DEBUG，避免淹没控制台"""
    _STATIC_EXTS = (".css", ".js", ".ico", ".png", ".jpg", ".jpeg",
                    ".gif", ".svg", ".woff", ".woff2", ".html", ".map")
    _API_PATHS = ("/chat", "/reset", "/health", "/stats", "/api/")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if any(p in msg for p in self._API_PATHS):
            return True
        if '"GET / HTTP' in msg or any(ext in msg for ext in self._STATIC_EXTS):
            record.levelno = logging.DEBUG
            record.levelname = "DEBUG"
        return True


logging.getLogger("werkzeug").addFilter(_StaticFilter())
logging.getLogger("dashscope").setLevel(logging.WARNING)   # 屏蔽 embedding 向量数组
logger = logging.getLogger(__name__)
# ──────────────────────────────────────────────────────────

# RAG服务（导入时临时屏蔽 stderr，避免 grpc/chromadb 初始化噪音）
_stderr = sys.stderr
try:
    sys.stderr = open(os.devnull, "w")
    from .core.rag_service import RAGService
finally:
    sys.stderr.close()
    sys.stderr = _stderr

from .core.persona_manager import PersonaManager

# 配置Flask应用，指定静态文件夹和模板文件夹
app = Flask(__name__,
            static_folder='front',
            static_url_path='')
CORS(app)  # 允许跨域访问

# 配置类
class Config:
    """应用配置"""
    # Qwen API配置
    QWEN_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode"
    QWEN_API_PATH = "/v1/chat/completions"
    CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen-plus")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v4")
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

    # RAG优化配置
    LLM_API_BASE = os.getenv("LLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode")
    LLM_REWRITING_MODEL = os.getenv("LLM_REWRITING_MODEL", "qwen-plus")
    RAG_QUERY_REWRITING_ENABLED = os.getenv("RAG_QUERY_REWRITING_ENABLED", "true").lower() == "true"
    RAG_COREFERENCE_RESOLUTION_ENABLED = os.getenv("RAG_COREFERENCE_RESOLUTION_ENABLED", "true").lower() == "true"

    # 生成参数
    TEMPERATURE = 0.5
    TOP_P = 0.7
    MAX_TOKENS = 500
    REPETITION_PENALTY = 1.2

    # RAG配置
    RAG_ENABLED = True
    RAG_MAX_RESULTS = 50
    RAG_MAX_CONTEXT_LENGTH = 2000
    RAG_INCLUDE_METADATA = True

    # ChromaDB配置（本地持久化，无需独立数据库服务）
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    # 系统提示前缀（RAG 注入固定格式）
    RAG_SYSTEM_PREFIX = "以下是与用户问题高度相关的历史聊天记录（若为空则表示未检索到）：\n"

    # 固定扮演指南（始终存在，无需用户写入）
    RAG_ROLE_INSTRUCTION = """

【回答要求】
1. 上方聊天记录是参考材料，用于理解被扮演者的说话风格、用词习惯、常见话题。
2. 你要扮演的是标签'发送者'的值为'self'的人，根据这些记录提炼出ta的语气特点（如口头禅、句子长短、表情包使用习惯），在回答中体现。
3. 将检索到的内容作为记忆背景，若问题与记录有关联，可自然融入，不要逐字引用。
4. 若检索结果为空或不相关，依据角色描述合理推断该人的说话方式作答。
5. 绝对不要提及'聊天记录'、'检索结果'、'向量数据库'、'系统提示'等技术词汇。
6. 回复长度和风格应与聊天记录中该人的习惯保持一致，避免过于正式或冗长。
7. 全程不使用任何表情符号、颜文字、emoji。
8. 不使用括号，不添加任何括号内的心理活动、动作、语气补充描写。
9. 只根据提供的参考聊天记录来模仿说话风格：
      句子长度、分段、语气、用词都尽量贴近参考记录。
      除非参考记录里本身就有表情、括号、心理描写，否则一律不主动添加。
      不额外发挥、不脑补情绪，只做自然、简洁、贴合原文风格的回复。
"""

    # Flask配置
    HOST = "0.0.0.0"
    PORT = 8080
    DEBUG = False


# 全局会话存储（简单演示用，生产环境建议使用Redis等）
sessions: Dict[str, List[Dict[str, str]]] = {}


class RAGServiceManager:
    """按需加载、缓存各分身的 RAGService"""

    def __init__(self):
        self._services: Dict[str, RAGService] = {}

    def get(self, persona: dict) -> Optional[RAGService]:
        pid = persona["id"]
        if pid not in self._services:
            try:
                svc = RAGService(
                    dashscope_api_key=Config.DASHSCOPE_API_KEY,
                    embed_model=Config.EMBED_MODEL,
                    collection_name=persona["collection"],
                    persist_directory=Config.CHROMA_PERSIST_DIR,
                    llm_api_base=Config.LLM_API_BASE,
                    llm_rewriting_model=Config.LLM_REWRITING_MODEL,
                    query_rewriting_enabled=Config.RAG_QUERY_REWRITING_ENABLED,
                    coreference_resolution_enabled=Config.RAG_COREFERENCE_RESOLUTION_ENABLED,
                )
                self._services[pid] = svc
                logger.info("RAGService 已为分身 '%s' 初始化（集合: %s）", persona["name"], persona["collection"])
            except Exception as e:
                logger.error("RAGService 初始化失败（分身: %s）: %s", persona["name"], e)
                return None
        return self._services[pid]

    def evict(self, persona_id: str):
        self._services.pop(persona_id, None)


# 全局实例
rag_manager = RAGServiceManager()
persona_manager = PersonaManager(Config.CHROMA_PERSIST_DIR)


def retrieve_rag_context(rag_service: RAGService, question: str, persona: Optional[dict] = None) -> Optional[str]:
    """从 RAGService 检索相关上下文"""
    if not rag_service or not Config.RAG_ENABLED:
        return None

    try:
        rp = (persona or {}).get("rag_params", {})
        results = rag_service.search(
            query=question,
            persona=persona,
            k=rp.get("k", Config.RAG_MAX_RESULTS),
            include_nearby=rp.get("include_nearby", True),
            time_window_minutes=rp.get("time_window_minutes", 30),
            nearby_per_result=rp.get("nearby_per_result", 8),
            max_total_results=rp.get("max_total_results", 50),
            lambda_mult=rp.get("lambda_mult", 0.6),
        )

        if not results:
            logger.info("  → RAG: 无检索结果")
            return None

        context = rag_service.format_context(
            results,
            max_context_length=Config.RAG_MAX_CONTEXT_LENGTH,
            include_metadata=Config.RAG_INCLUDE_METADATA
        )

        if not context:
            logger.info("  → RAG: 格式化上下文为空")
            return None

        logger.info("  → RAG: 注入 %d 条聊天记录到提示词", len(results))
        return Config.RAG_SYSTEM_PREFIX + "\n" + context

    except Exception as e:
        logger.warning("RAG检索异常: %s", e, exc_info=True)
        return None


def inject_rag_context(messages: List[Dict[str, str]], rag_text: str, system_prompt: str) -> List[Dict[str, str]]:
    """在消息列表开头注入 RAG 上下文 + 分身系统提示词 + 固定扮演指南"""
    injected = list(messages)
    combined = f"{rag_text}\n\n{system_prompt}{Config.RAG_ROLE_INSTRUCTION}"
    injected.insert(0, {"role": "system", "content": combined})
    return injected


def call_qwen_api(messages: List[Dict[str, str]], max_tokens: int = Config.MAX_TOKENS) -> Tuple[Optional[str], Optional[str]]:
    """调用Qwen API"""
    endpoint = f"{Config.QWEN_API_BASE.rstrip('/')}/{Config.QWEN_API_PATH.lstrip('/')}"

    payload = {
        "model": Config.CHAT_MODEL,
        "messages": messages,
        "temperature": Config.TEMPERATURE,
        "top_p": Config.TOP_P,
        "max_tokens": max_tokens,
        "repetition_penalty": Config.REPETITION_PENALTY,
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Config.DASHSCOPE_API_KEY}"
    }

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            return None, f"API返回错误: {response.status_code} - {response.text}"

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return None, "API返回结果为空"

        content = choices[0].get("message", {}).get("content", "")
        return content, None

    except requests.RequestException as e:
        return None, f"API请求失败: {str(e)}"
    except Exception as e:
        return None, f"处理响应时出错: {str(e)}"


@app.route("/")
def index():
    """首页 - 返回前端页面"""
    return send_from_directory('front', 'index.html')


@app.route("/api/personas", methods=["GET"])
def list_personas():
    """列出所有分身"""
    personas = persona_manager.list()
    return jsonify({"status": "success", "personas": personas})


@app.route("/api/personas/<persona_id>", methods=["DELETE"])
def delete_persona(persona_id: str):
    """删除分身记录（不删除 ChromaDB 数据）"""
    deleted = persona_manager.delete(persona_id)
    if not deleted:
        return jsonify({"status": "error", "error": "分身不存在"}), 404
    rag_manager.evict(persona_id)
    return jsonify({"status": "success", "message": "分身已删除"})


@app.route("/chat", methods=["POST"])
def chat():
    """处理聊天请求"""
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        session_id = data.get("session_id", "default")
        persona_id = data.get("persona_id")

        if not user_message:
            return jsonify({
                "status": "error",
                "error": "消息不能为空"
            }), 400

        # 获取分身信息（无 persona_id 时回落到第一个分身）
        persona = None
        if persona_id:
            persona = persona_manager.get(persona_id)
        if persona is None:
            all_personas = persona_manager.list()
            persona = all_personas[0] if all_personas else None

        # 获取对应的 RAGService（懒加载）
        rag_service = rag_manager.get(persona) if persona else None
        system_prompt = persona["system_prompt"] if persona else (
            "你是一个智能助手，用自然、口语化、简洁的中文回答。"
        )

        # 获取或创建会话
        if session_id not in sessions:
            sessions[session_id] = []

        messages = sessions[session_id]

        # 添加用户消息
        messages.append({"role": "user", "content": user_message})

        # 检索RAG上下文
        messages_for_call = list(messages)
        rag_context = None
        if Config.RAG_ENABLED and rag_service:
            rag_context = retrieve_rag_context(rag_service, user_message, persona)
            if rag_context:
                messages_for_call = inject_rag_context(messages_for_call, rag_context, system_prompt)

        # 调用Qwen API
        max_tokens = persona.get("model_params", {}).get("max_tokens", Config.MAX_TOKENS) if persona else Config.MAX_TOKENS
        reply, error = call_qwen_api(messages_for_call, max_tokens=max_tokens)

        if error:
            return jsonify({
                "status": "error",
                "error": error
            }), 500

        # 保存助手回复
        messages.append({"role": "assistant", "content": reply})

        # 对话链路日志（DEBUG 级别，写入文件）
        logger.debug(
            "[对话] session=%s persona=%s\n>>> 用户: %s\n--- RAG检索结果 ---\n%s\n--- 数字分身回复 ---\n%s",
            session_id, persona["name"] if persona else "无", user_message, rag_context or "(无)", reply,
        )

        # 限制会话长度（保留最近20轮对话）
        if len(messages) > 40:
            messages[:] = messages[-40:]

        return jsonify({
            "status": "success",
            "reply": reply,
            "session_id": session_id
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route("/reset", methods=["POST"])
def reset_session():
    """重置会话"""
    data = request.get_json() or {}
    session_id = data.get("session_id", "default")

    if session_id in sessions:
        del sessions[session_id]

    return jsonify({
        "status": "success",
        "message": "会话已重置"
    })


@app.route("/health", methods=["GET"])
def health():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "rag_enabled": Config.RAG_ENABLED,
        "persona_count": len(persona_manager.list()),
    })


@app.route("/stats", methods=["GET"])
def stats():
    """获取所有分身的 RAG 数据库统计信息"""
    personas = persona_manager.list()
    result = []
    for p in personas:
        svc = rag_manager.get(p)
        if svc:
            try:
                s = svc.get_stats()
                result.append({"persona": p["name"], "stats": s})
            except Exception as e:
                result.append({"persona": p["name"], "error": str(e)})
    return jsonify({"status": "success", "data": result})


if __name__ == "__main__":
    logger.info("Flask聊天机器人服务器启动中...")
    logger.info("本地访问: http://localhost:%s", Config.PORT)
    logger.info("局域网访问: http://0.0.0.0:%s", Config.PORT)
    logger.info("RAG状态: %s", "已启用" if Config.RAG_ENABLED else "未启用")
    personas = persona_manager.list()
    logger.info("已加载分身: %d 个", len(personas))
    for p in personas:
        logger.info("  - %s（集合: %s，%d 条记录）", p["name"], p["collection"], p["doc_count"])
    logger.info("日志级别: %s  日志文件: %s", os.getenv("LOG_LEVEL", "INFO"), _LOG_FILE)

    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
