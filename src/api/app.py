"""
Flask 聊天服务器主程序
"""

import os
# 抑制 gRPC 重复注册指标的告警（chromadb 和 OpenTelemetry 同时使用 gRPC 导致）
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import logging
from dotenv import load_dotenv

# 必须在 blueprint / telemetry 导入之前完成日志配置
# 否则 telemetry 初始化会先往 root logger 添加 OTel handler，
# 导致 basicConfig 检测到已有 handler 而整个跳过，root level 停留在 WARNING
load_dotenv()

# 根日志级别用 LOG_LEVEL（DEBUG），确保 OTel handler 能收到 DEBUG 日志发往 Loki
# 控制台单独用 CONSOLE_LOG_LEVEL（默认 INFO），终端只看关键信息
_root_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
_console_level = getattr(logging, os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper(), logging.INFO)

root_logger = logging.getLogger()
root_logger.setLevel(_root_level)

console_handler = logging.StreamHandler()
console_handler.setLevel(_console_level)
console_handler.setFormatter(logging.Formatter(
    "[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
root_logger.addHandler(console_handler)

# 抑制 urllib3 连接池噪音日志（监控组件的 HTTP 请求）
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
# 抑制 DashScope/httpx SDK 的 debug 日志（embedding 向量矩阵等）
logging.getLogger("dashscope").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from flask import Flask, send_from_directory
from flask_cors import CORS
from src.api.config import Config
from src.api.routes.chatbot import chat_bp
from src.api.routes.persona import persona_bp
from src.api.routes.tutor import tutor_bp

logger = logging.getLogger(__name__)


def create_app():
    """初始化并配置 Flask 应用"""
    # 静态文件夹路径
    static_folder = os.path.join(Config.PROJECT_ROOT, "frontend")
    
    app = Flask(__name__, static_folder=static_folder, static_url_path="")
    
    if Config.CORS_ENABLED:
        CORS(app)
        logger.info("CORS 已启用")

    # 注册各个功能的路由 Blueprint
    app.register_blueprint(chat_bp)
    app.register_blueprint(persona_bp)
    app.register_blueprint(tutor_bp, url_prefix="") # 助教接口在原来的基础上可能带 /tutor 路径

    @app.route("/")
    def index():
        """根路径返回对话主页"""
        return send_from_directory(static_folder, "index.html")

    @app.route("/tutor")
    def tutor():
        """返回数字助教页"""
        return send_from_directory(static_folder, "tutor.html")

    @app.route("/exports/<path:filename>")
    def exported_assets(filename):
        """提供导出图片等静态资源"""
        return send_from_directory(Config.PDF_EXPORT_ROOT, filename)

    return app


if __name__ == "__main__":
    logger.info("DigitalTwin-Refactor API Server 启动中...")

    base_url = f"http://localhost:{Config.PORT}"
    logger.info(f"数字分身: {base_url}")
    logger.info(f"数字助教: {base_url}/tutor")

    app = create_app()
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
