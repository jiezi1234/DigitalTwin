"""
Self-RAG 反思与质量评估模块
通过内联反思标签评估检索必要性、内容相关性及回复可靠性。
"""

import re
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

# ── 标签正则 ────────────────────────────────────────────
RE_RETRIEVE  = re.compile(r'\[Retrieve\]\s*(是|否)')
RE_ISREL     = re.compile(r'\[IsRel\]\s*(\d+)\s*(相关|不相关)')
RE_ISSUP     = re.compile(r'\[IsSup\]\s*(完全支持|部分支持|无支持)')
RE_ISUSE     = re.compile(r'\[IsUse\]\s*(\d)')
RE_ANY_TOKEN = re.compile(r'\[(?:Retrieve|IsRel|IsSup|IsUse)\][^\n]*(?:\n|$)')


class SelfRAG:
    """Self-RAG 反思增强组件"""

    def __init__(
        self,
        llm_client: LLMClient,
        thresholds: Optional[Dict[str, Any]] = None,
        prompts: Optional[Dict[str, str]] = None
    ):
        """
        初始化 Self-RAG 组件
        
        Args:
            llm_client: 用于反思任务的 LLM 客户端
            thresholds: 评估阈值 (relevance, support, utility)
            prompts: 自定义提示词模板
        """
        self.llm_client = llm_client
        self.thresholds = thresholds or {
            "relevance": 0.5,
            "support": 0.5,
            "utility": 3
        }
        self.prompts = prompts or {}

    # 默认检索决策提示词（聊天记录场景）
    DEFAULT_DECIDE_PROMPT = (
        "判断以下用户消息是否需要从历史聊天记录中检索信息来回答。\n"
        "不需要检索的情况：简单问候、询问身份、对话延续、通用知识问题。\n"
        "需要检索的情况：询问某人说过什么、特定事件或话题、涉及聊天记录中的人物或事件。\n\n"
        "用户消息：{query}\n\n"
        "请用以下格式回答（只输出这一行）：\n"
        "[Retrieve] 是 或 [Retrieve] 否"
    )

    # 知识库检索决策提示词（教材/文档场景）
    KNOWLEDGE_DECIDE_PROMPT = (
        "判断以下用户消息是否需要从知识库中检索信息来回答。\n"
        "不需要检索的情况：简单问候、闲聊。\n"
        "需要检索的情况：涉及专业知识、概念解释、技术问题、课程内容。\n\n"
        "用户消息：{query}\n\n"
        "请用以下格式回答（只输出这一行）：\n"
        "[Retrieve] 是 或 [Retrieve] 否"
    )

    def decide_retrieval(self, query: str) -> bool:
        """判断是否需要检索"""
        with tracer.start_as_current_span("selfrag.decide_retrieval") as span:
            template = self.prompts.get("decide_retrieval", self.DEFAULT_DECIDE_PROMPT)
            prompt = template.format(query=query)
            
            result = self.llm_client.call([{"role": "user", "content": prompt}], temperature=0.3)
            if not result:
                logger.warning("Self-RAG 决策调用失败，默认开启检索")
                return True

            m = RE_RETRIEVE.search(result)
            need = (m.group(1) == "是") if m else ("是" in result)

            span.set_attribute("selfrag.need_retrieval", need)
            logger.debug(f"[Self-RAG 检索决策] query={query} | need={need} | raw={result.strip()}")
            return need

    def evaluate_relevance(self, query: str, results: List[Tuple[str, Dict[str, Any], float]]) -> Dict[int, bool]:
        """评估检索到的段落相关性"""
        if not results:
            return {}

        with tracer.start_as_current_span("selfrag.evaluate_relevance") as span:
            # 构建编号段落
            passages = []
            for i, (content, _, _) in enumerate(results[:15], 1):
                preview = content.strip().replace("\n", " ")[:120]
                passages.append(f"[{i}] {preview}")
            
            passages_text = "\n".join(passages)
            
            prompt = (
                "判断以下检索到的聊天记录段落是否与用户问题相关。\n\n"
                f"检索段落：\n{passages_text}\n\n"
                "【重要】请严格按照下面格式逐行输出，不要解释：\n"
                "[IsRel] 1 相关\n"
                "[IsRel] 2 不相关\n"
                "..."
            )
            
            raw = self.llm_client.call([
                {"role": "system", "content": "你是一个相关性判断助手。"},
                {"role": "user", "content": f"问题：{query}\n\n{prompt}"}
            ], temperature=0.1)
            
            is_rel_map = {}
            if raw:
                for m in RE_ISREL.finditer(raw):
                    is_rel_map[int(m.group(1))] = (m.group(2) == "相关")
            
            # 补齐未解析到的部分，默认相关
            for i in range(1, len(results[:15]) + 1):
                if i not in is_rel_map:
                    is_rel_map[i] = True
            
            logger.debug(f"[Self-RAG 相关性] {sum(is_rel_map.values())}/{len(is_rel_map)} 相关 | {is_rel_map}")
            return is_rel_map

    def critique_output(self, text: str) -> Tuple[str, str, int]:
        """解析输出中的反思标签 [IsSup] 和 [IsUse]"""
        is_sup = "部分支持"
        is_use = 3
        
        m_sup = RE_ISSUP.search(text)
        if m_sup:
            is_sup = m_sup.group(1)
            
        m_use = RE_ISUSE.search(text)
        if m_use:
            is_use = int(m_use.group(1))
            
        # 移除所有标签得到纯回复
        clean_text = RE_ANY_TOKEN.sub("", text).strip()
        logger.debug(f"[Self-RAG 评价] IsSup={is_sup} | IsUse={is_use}")
        return clean_text, is_sup, is_use

    @staticmethod
    def get_critique_instruction() -> str:
        """获取注入到 Prompt 中的自我评价指令"""
        return (
            "\n\n【Self-RAG 反思要求】\n"
            "在回复结束后的最后一行，请换行输出以下两个评估标记：\n"
            "[IsSup] 完全支持/部分支持/无支持 (评估你的回复是否被上方检索到的事实所支持)\n"
            "[IsUse] 1-5 (评估你的回复对用户的有用程度，5分最有用)\n"
        )
