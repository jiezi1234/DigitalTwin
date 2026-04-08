from src.skills.base_skill.scripts.base_skill import BaseSkill
import logging
logger = logging.getLogger(__name__)

class PDFAssetsExportSkill(BaseSkill):
    name = "export_pdf_assets"
    description = (
        "导出给定 PDF 文件的基础文字和图块为中间商结构包。"
    )

    def run(self, action_input: str) -> str:
        logger.info(f"[PDFAssetsExportSkill] 触发 PDF 提取，参数: {action_input}")
        return "提交了导出任务，将在 output/ 产生抽帧结果。"
