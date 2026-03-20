import os
# 抑制 gRPC 重复注册指标的告警
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import logging
from src.api.app import create_app
from src.api.config import Config

logger = logging.getLogger(__name__)

def main():
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

if __name__ == "__main__":
    main()
