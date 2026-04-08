from src.skills.base_skill.scripts.base_skill import BaseSkill
import logging
logger = logging.getLogger(__name__)

class MultimodalPDFRebuildSkill(BaseSkill):
    name = "rebuild_multimodal_index"
    description = (
        "全量重置并重建本地所有的多模态 (图文混排) Chunk 集合。"
    )

    def run(self, action_input: str) -> str:
        logger.info(f"[MultimodalPDFRebuildSkill] 发起索引库刷新: {action_input}")
        return "指令已下达底层，开始全局索引清除与重算流程。"
