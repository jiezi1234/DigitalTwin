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
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v4")
    MM_EMBED_MODEL = os.getenv("MM_EMBED_MODEL", "multimodal-embedding-v1")
    
    # RAG 参数
    RAG_MAX_RESULTS = int(os.getenv("RAG_MAX_RESULTS", "50"))
    RAG_MAX_CONTEXT_LENGTH = int(os.getenv("RAG_MAX_CONTEXT_LENGTH", "2000"))
    
    # 指令集 (从原 app.py 迁移)
    RAG_SYSTEM_PREFIX = "以下是与用户问题高度相关的历史聊天记录（若为空则表示未检索到）：\n"
    
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

    # 数字助教配置
    TUTOR_ENABLED = os.getenv("TUTOR_ENABLED", "true").lower() == "true"
    TUTOR_COLLECTION = os.getenv("TUTOR_COLLECTION", "textbook_embeddings")
    TUTOR_MM_TEXT_COLLECTION = os.getenv("TUTOR_MM_TEXT_COLLECTION", "textbook_mm_text_embeddings")
    TUTOR_MM_IMAGE_COLLECTION = os.getenv("TUTOR_MM_IMAGE_COLLECTION", "textbook_mm_image_embeddings")
    TUTOR_OCR_TEXT_COLLECTION = os.getenv("TUTOR_OCR_TEXT_COLLECTION", "textbook_ocr_text_embeddings")
    PDF_EXPORT_ROOT = os.path.abspath(
        os.getenv("PDF_EXPORT_ROOT", os.path.join(PROJECT_ROOT, "output"))
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
