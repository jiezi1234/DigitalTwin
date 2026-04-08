"""
Web API 配置管理
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """API 全局配置"""
    
    # 基础路径
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", os.path.join(PROJECT_ROOT, "chroma_db"))
    
    # LLM 配置
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    LLM_API_BASE = os.getenv("LLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen-plus")
    TUTOR_VL_MODEL = os.getenv("TUTOR_VL_MODEL", "qwen-vl-plus")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v2")
    MM_EMBED_MODEL = os.getenv("MM_EMBED_MODEL", "multimodal-embedding-v1")
    
    # RAG 参数
    RAG_MAX_RESULTS = int(os.getenv("RAG_MAX_RESULTS", "50"))
    RAG_MAX_CONTEXT_LENGTH = int(os.getenv("RAG_MAX_CONTEXT_LENGTH", "2000"))
    
    # 数字助教配置
    TUTOR_ENABLED = os.getenv("TUTOR_ENABLED", "true").lower() == "true"
    TUTOR_COLLECTION = os.getenv("TUTOR_COLLECTION", "textbook_embeddings")
    TUTOR_MM_TEXT_COLLECTION = os.getenv("TUTOR_MM_TEXT_COLLECTION", "textbook_mm_text_embeddings")
    TUTOR_MM_IMAGE_COLLECTION = os.getenv("TUTOR_MM_IMAGE_COLLECTION", "textbook_mm_image_embeddings")
    TUTOR_OCR_TEXT_COLLECTION = os.getenv("TUTOR_OCR_TEXT_COLLECTION", "textbook_ocr_text_embeddings")
    PDF_EXPORT_ROOT = os.path.abspath(
        os.getenv("PDF_EXPORT_ROOT", os.path.join(PROJECT_ROOT, "output", "course_mm"))
    )
    TUTOR_MAX_CONTEXT_LENGTH = int(os.getenv("TUTOR_MAX_CONTEXT_LENGTH", "4000"))
    TUTOR_MAX_TOKENS = int(os.getenv("TUTOR_MAX_TOKENS", "1500"))
    TUTOR_SYSTEM_PROMPT = os.getenv("TUTOR_SYSTEM_PROMPT", """你是一位数据库课程的数字助教。你的职责是基于课本内容帮助学生理解数据库相关知识。

【回答要求】
1. 基于提供的课本内容准确回答学生问题。
2. 回答要清晰、易懂，适当使用示例和类比帮助理解。
3. 对于SQL相关问题，提供具体的SQL语句示例。
4. 在回答中引用课本内容时，使用对应的编号标注，如 [1][3]。只引用你实际使用的段落编号。
5. 如果课本内容中没有相关信息，诚实说明并尝试基于数据库通用知识回答。
6. 鼓励学生思考，可以适当提出引导性问题。
7. 使用中文回答。""")

    # Flask 配置
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8080"))
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    CORS_ENABLED = True
