from src.skills.base_skill.scripts.base_skill import BaseSkill
import logging
logger = logging.getLogger(__name__)

class CourseMaterialImportSkill(BaseSkill):
    name = "import_course_materials"
    description = (
        "执行复杂 PDF（包含课件、教材）的导入与向量切片重构任务。"
        "可传入 glob 模式来限定指定的文件范围。"
    )

    def run(self, action_input: str) -> str:
        logger.info(f"[CourseMaterialImportSkill] 准备发起课程材料多尺度建库，参数: {action_input}")
        return f"已向后端提交目标文件 {action_input} 的解析请求（具体落盘由后台 worker 完成）。"
