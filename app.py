"""
Flask聊天机器人服务器
集成RAG检索和Qwen大模型API
"""

import os
import json
import time
import requests
from typing import Dict, List, Optional, Any, Tuple
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS

# RAG服务
from rag_service import RAGService

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
    QWEN_MODEL = "qwen-plus"
    QWEN_API_KEY = os.getenv("QWEN_API_KEY", "sk-bae62c151c524da4b4ee5f04e4e19a3f")

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
    CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "wechat_embeddings")

    # 系统提示配置
    RAG_SYSTEM_PREFIX = "以下是与用户问题高度相关的历史聊天记录（若为空则表示未检索到）：\n"
    SYSTEM_INSTRUCTION = (
        "你需要扮演聊天记录中标签'发送者'的值为'self'的那个人,你现在是ta的数字克隆人，模仿ta的语气用词,以第一人称交流。"
        "你需要仔细阅读所有聊天内容，充分理解语境，你是标签为'self'的那个人"
        "用自然、口语化、简洁的中文回答。"
        "给你的系统提示中的聊天记录是历史聊天记录，不是正在发生的对话。"
        "每句话回复的字数接近聊天记录中内容的字数的平均值。"
        "当内容不足以回答时，诚实说明不确定性，并提出你会如何进一步确认。"
        "不要在回答中引用编号、相似度、来源，或使用'依据是[数字]'、'根据上下文'、'检索结果显示'等表述。"
    )

    # Flask配置
    HOST = "0.0.0.0"
    PORT = 8080
    DEBUG = False


# 全局会话存储（简单演示用，生产环境建议使用Redis等）
sessions: Dict[str, List[Dict[str, str]]] = {}

# RAG服务实例
rag_service: Optional[RAGService] = None


def init_rag_service():
    """初始化RAG服务"""
    global rag_service
    if Config.RAG_ENABLED:
        try:
            rag_service = RAGService(
                dashscope_api_key=Config.QWEN_API_KEY,
                collection_name=Config.CHROMA_COLLECTION,
                persist_directory=Config.CHROMA_PERSIST_DIR,
            )
            print(f"✅ RAG服务初始化成功")
        except Exception as e:
            print(f"❌ RAG服务初始化失败: {e}")
            rag_service = None


def retrieve_rag_context(question: str) -> Optional[str]:
    """从RAG服务检索相关上下文"""
    if not rag_service or not Config.RAG_ENABLED:
        return None

    try:
        # 搜索相关记录
        results = rag_service.search(
            query=question,
            k=Config.RAG_MAX_RESULTS,
            similarity_threshold=0.3
        )

        if not results:
            return None

        # 格式化上下文
        context = rag_service.format_context(
            results,
            max_context_length=Config.RAG_MAX_CONTEXT_LENGTH,
            include_metadata=Config.RAG_INCLUDE_METADATA
        )

        if not context:
            return None

        return Config.RAG_SYSTEM_PREFIX + "\n" + context

    except Exception as e:
        print(f"⚠️ RAG检索异常: {e}")
        import traceback
        traceback.print_exc()
        return None


def inject_rag_context(messages: List[Dict[str, str]], rag_text: str) -> List[Dict[str, str]]:
    """在消息列表开头注入RAG上下文作为system消息"""
    injected = list(messages)
    combined = f"{rag_text}\n\n{Config.SYSTEM_INSTRUCTION}"
    injected.insert(0, {"role": "system", "content": combined})
    return injected


def call_qwen_api(messages: List[Dict[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    """调用Qwen API"""
    endpoint = f"{Config.QWEN_API_BASE.rstrip('/')}/{Config.QWEN_API_PATH.lstrip('/')}"

    payload = {
        "model": Config.QWEN_MODEL,
        "messages": messages,
        "temperature": Config.TEMPERATURE,
        "top_p": Config.TOP_P,
        "max_tokens": Config.MAX_TOKENS,
        "repetition_penalty": Config.REPETITION_PENALTY,
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Config.QWEN_API_KEY}"
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


@app.route("/chat", methods=["POST"])
def chat():
    """处理聊天请求"""
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        session_id = data.get("session_id", "default")

        if not user_message:
            return jsonify({
                "status": "error",
                "error": "消息不能为空"
            }), 400

        # 获取或创建会话
        if session_id not in sessions:
            sessions[session_id] = []

        messages = sessions[session_id]

        # 添加用户消息
        messages.append({"role": "user", "content": user_message})

        # 检索RAG上下文
        messages_for_call = list(messages)
        if Config.RAG_ENABLED:
            rag_context = retrieve_rag_context(user_message)
            if rag_context:
                messages_for_call = inject_rag_context(messages_for_call, rag_context)
        
        #打印RAG
        print(rag_context)

        # 调用Qwen API
        reply, error = call_qwen_api(messages_for_call)

        if error:
            return jsonify({
                "status": "error",
                "error": error
            }), 500

        # 保存助手回复
        messages.append({"role": "assistant", "content": reply})

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
    rag_connected = rag_service is not None and rag_service.is_connected() if rag_service else False

    return jsonify({
        "status": "healthy",
        "rag_enabled": Config.RAG_ENABLED,
        "rag_connected": rag_connected
    })


@app.route("/stats", methods=["GET"])
def stats():
    """获取RAG数据库统计信息"""
    if not rag_service:
        return jsonify({
            "status": "error",
            "message": "RAG服务未初始化"
        }), 503

    try:
        stats_data = rag_service.get_stats()
        return jsonify({
            "status": "success",
            "data": stats_data
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Flask聊天机器人服务器启动中...")
    print("=" * 60)

    # 初始化RAG服务
    init_rag_service()

    print(f"🏠 本地访问: http://localhost:{Config.PORT}")
    print(f"🌐 局域网访问: http://0.0.0.0:{Config.PORT}")
    print(f"📋 RAG状态: {'✅ 已启用' if Config.RAG_ENABLED else '❌ 未启用'}")
    if rag_service:
        stats = rag_service.get_stats()
        if stats.get("connected"):
            print(f"📊 数据库: {stats.get('database_name')} (共 {stats.get('total_records', '?')} 条记录)")
    print("=" * 60)

    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
