from src.skills.base_skill.scripts.base_skill import BaseSkill
import logging
logger = logging.getLogger(__name__)

class WeChatCsvImportSkill(BaseSkill):
    name = "import_wechat_csv"
    description = (
        "将 CSV 格式的微信聊天记录导入系统知识库。"
        "通常在添加新语料库或更新记忆库时被触发。不需要输入特定参数。"
    )

    def run(self, action_input: str) -> str:
        # 该导入目前属于重型后台任务，通常需要交互式选取 Persona 等，
        # 在面向 Agent 时可做封装静默执行。此处保留接口占位。
        logger.info(f"[WeChatCsvImportSkill] 触发自动导入: {action_input}")
        return "代理式后台录入已排队，请在后台终端提供精确的解析目录参数。"
