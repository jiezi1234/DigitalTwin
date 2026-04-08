"""
数字助教专用 ReAct 智能体 (TutorReActAgent)

与 ReActAgent（分身智能体）的关键区别：
- 不含 persona（助教没有人格设定，由系统 prompt 定义角色）
- run() 返回 (reply_text, image_hits, text_results) 三元组
  image_hits: List[Dict] — 命中的图片元数据，供路由层推流
  text_results: List     — 命中的文字段落原始数据，供路由层提取sources
"""

import re
import logging
from typing import List, Dict, Any, Tuple

from src.api.config import Config
from src.infrastructure.llm_client import LLMClient
from src.skills.base_skill.scripts.base_skill import BaseSkill

logger = logging.getLogger(__name__)


TUTOR_REACT_SYSTEM_PROMPT = """你是一位数据库课程的数字助教，具备主动检索课本资料的能力。

你可以使用以下工具：
{tools_desc}

你必须严格遵循以下 ReAct 格式，每次只输出一步，然后等待系统响应：

第一步 — 如果需要查询工具：
Thought: 分析学生的问题，说明需要调用哪个工具以及原因
Action: 工具名称（仅限 {tool_names}）
Action Input: 传给工具的查询词

⚠️ 写完 Action Input 后，立即停止输出！不要自己写 Observation，不要自己猜测结果，不要写 Final Answer！等待系统填充 Observation 后你才能继续。

第二步 — 收到 Observation 后：
Thought: 分析检索结果，决定是否需要继续检索或给出最终答案
Final Answer: 基于检索结果给出的完整中文解答（在正文中用 [1][2] 标注引用编号）

第三步 — 如果不需要工具（简单问候/非专业问题）：
Thought: 分析后判断不需要检索课本
Action: None
Action Input: None
Final Answer: 直接回答

规则：
1. 对于概念、定义、SQL语法、理论知识 → 调用 search_textbook_text
2. 用户要求看图/展示图表 → 调用 search_textbook_images
3. 只要是“课程相关事实问题”（包括课程介绍、成绩构成、考试比例、学时、实验安排、教材、章节内容），必须先调用至少一个工具，再给 Final Answer。
4. 不允许用“这通常不在课本里”作为跳过检索的理由。先检索，再判断是否缺失。
5. 绝对不要在 Final Answer 末尾写"参考资料"或来源文件名，系统会自动渲染引用区域
6. 必须用中文回答
"""

RE_INTERNAL_REASONING_PREFIX = re.compile(
    r"^(Thought|思考|分析|推理|理由|依据|从检索结果来看)\s*[:：]\s*",
    re.IGNORECASE,
)
RE_INTERNAL_REASONING_LINE = re.compile(
    r"^\s*(Thought|思考|分析|推理|理由|依据|从检索结果来看)\s*[:：].*$",
    re.IGNORECASE,
)


class TutorReActAgent:
    """数字助教专用 ReAct 智能体"""

    def __init__(
        self,
        llm_client: LLMClient,
        tools: List[BaseSkill],
        max_iterations: int = 5,
    ):
        self.llm_client = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations

    @staticmethod
    def _build_text_context(
        text_results: List[Any],
        max_items: int = 12,
        max_chars_per_item: int = 800,
    ) -> str:
        if not text_results:
            return ""
        lines: List[str] = []
        for idx, (content, metadata, _) in enumerate(text_results[:max_items], 1):
            source_file = metadata.get("source_file", "")
            page = metadata.get("page", "")
            location = " > ".join(
                [p for p in [source_file, f"第{page}页" if page else ""] if p]
            )
            text = (content or "").strip().replace("\n", " ")
            if len(text) > max_chars_per_item:
                text = text[:max_chars_per_item] + "..."
            lines.append(f"[{idx}]【{location}】\n{text}")
        return "\n\n".join(lines)

    def _humanize_answer(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
        text_results: List[Any],
        image_hits: List[Dict[str, Any]],
        draft_answer: str,
        model: str,
        max_tokens: int,
    ) -> str:
        """
        第二阶段生成：
        使用更教学化的人设 prompt，在“已有检索证据”基础上给出自然回答。
        """
        context = self._build_text_context(text_results=text_results)
        image_refs = ", ".join([img.get("image_ref", "") for img in image_hits[:5] if img.get("image_ref")])
        if not image_refs:
            image_refs = "无"

        system_prompt = (
            f"{Config.TUTOR_SYSTEM_PROMPT}\n\n"
            "你正在进行最终回答润色。请注意：\n"
            "1. 语气自然、像真人助教，先直答再简短解释，不要机械模板。\n"
            "2. 若证据中有明确比例/数字，优先直接给出。\n"
            "3. 仅使用已给证据回答；若证据不足，明确指出不足。\n"
            "4. 正文中保留引用编号格式 [1][2]；不要输出“参考资料”标题。\n"
            "5. 严禁输出你的思考过程、分析过程或检索过程描述。\n"
            "6. 禁止出现“分析：”“从检索结果来看”“Thought:”这类字样。\n"
            "7. 只输出给学生看的最终回答正文，使用中文。"
        )
        user_prompt = (
            f"学生问题：{query}\n\n"
            f"检索证据：\n{context or '（无）'}\n\n"
            f"可用图片引用：{image_refs}\n\n"
            f"Agent 草稿回答：{draft_answer or '（无）'}\n\n"
            "请输出最终回答："
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history[-6:])
        messages.append({"role": "user", "content": user_prompt})

        refined = self.llm_client.call(
            messages=messages,
            temperature=0.6,
            max_tokens=max_tokens,
            model=model,
        )
        if refined:
            logger.info("[Tutor Agent] 二阶段生成完成（humanized answer）。")
            refined = refined.strip()
            refined = re.sub(r'^Final Answer:\s*', '', refined)
            refined = self._strip_internal_reasoning(refined)
            return refined
        fallback = draft_answer or "抱歉，我暂时无法组织回答，请稍后再试。"
        return self._strip_internal_reasoning(fallback)

    @staticmethod
    def _strip_internal_reasoning(text: str) -> str:
        if not text:
            return text
        lines = text.splitlines()
        cleaned: List[str] = []
        for line in lines:
            if RE_INTERNAL_REASONING_LINE.match(line.strip()):
                continue
            cleaned.append(line)
        result = "\n".join(cleaned).strip()
        # 若首行仍带前缀，做一次前缀剥离
        result = RE_INTERNAL_REASONING_PREFIX.sub("", result).strip()
        return result

    def run(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
        max_tokens: int = 1500,
        model: str = "qwen-vl-plus",
    ) -> Tuple[str, List[Dict[str, Any]], List[Any]]:
        """
        运行助教 ReAct 主循环

        Returns:
            reply_text: 最终回答文字
            image_hits: 所有本次对话命中的图片序列化列表
            text_results: 文字检索的原始结果（供提取 sources）
        """
        tools_desc = "\n".join([f"- {name}: {t.description}" for name, t in self.tools.items()])
        tool_names = ", ".join(self.tools.keys())

        system_prompt = TUTOR_REACT_SYSTEM_PROMPT.format(
            tools_desc=tools_desc,
            tool_names=tool_names,
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history[-10:])

        react_context = f"Question: {query}\n"

        # 收集本轮所有工具输出
        accumulated_image_hits: List[Dict[str, Any]] = []
        accumulated_text_results: List[Any] = []
        has_tool_call = False
        draft_final_answer = ""

        for i in range(self.max_iterations):
            current_messages = list(messages)
            current_messages.append({"role": "user", "content": react_context})
            consecutive_tool_failures = getattr(self, "_consecutive_failures", 0)

            response_text = self.llm_client.call(
                current_messages,
                temperature=0.3,
                max_tokens=800,
                model=model,
            )

            if not response_text:
                logger.error("[Tutor Agent] LLM 返回空。")
                return "抱歉，我暂时无法思考和回答，请稍后再试。", accumulated_image_hits, accumulated_text_results

            logger.info(
                f"========== [Tutor Agent] 迭代回合 {i+1} ==========\n"
                f"模型内部思考与决策:\n{response_text}\n"
                f"=================================================="
            )

            # 解析 Action
            action_match = re.search(r"Action:\s*(.*?)\n", response_text + "\n")
            action_input_match = re.search(r"Action Input:\s*(.*?)\n", response_text + "\n")

            if action_match and action_input_match:
                action = action_match.group(1).strip()
                action_input = action_input_match.group(1).strip()

                if action and action not in ("None", "none", ""):
                    # 严格截断回合内容，防止 LLM 幻觉自填 Observation
                    clean_match = re.search(r"(.*?Action Input:.*?\n)", response_text + "\n", re.DOTALL)
                    react_context += clean_match.group(1) if clean_match else response_text

                    if action not in self.tools:
                        logger.warning(f"[Tutor Agent] 未知技能: {action}")
                        react_context += f"Observation: 技能 {action} 不存在，可用技能: {tool_names}\n"
                        continue

                    logger.info(
                        f"\n>>>> [Tutor Agent] 开始调用外部技能 <<<<\n"
                        f"技能名称: {action}\n传入参数: {action_input}"
                    )
                    tool = self.tools[action]
                    observation = tool.run(action_input)
                    has_tool_call = True

                    # 收集图片和文字结果
                    from src.skills.textbook_retrieval.scripts.textbook_image_skill import TextbookImageSkill
                    from src.skills.textbook_retrieval.scripts.textbook_text_skill import TextbookTextSkill
                    if isinstance(tool, TextbookImageSkill):
                        accumulated_image_hits.extend(tool._last_image_results)
                    elif isinstance(tool, TextbookTextSkill):
                        accumulated_text_results.extend(tool._last_results)

                    logger.info(
                        f"<<<< [Tutor Agent] 技能调用结束，返回数据 >>>>\n{observation}\n" + "-" * 50
                    )

                    # 检测工具是否返回了错误（403/500/异常等）
                    is_tool_error = any(kw in observation for kw in [
                        "异常", "Error", "error", "403", "500", "failed", "失败"
                    ])
                    if is_tool_error:
                        self._consecutive_failures = getattr(self, "_consecutive_failures", 0) + 1
                        logger.warning(f"[Tutor Agent] 工具调用失败 (连续第 {self._consecutive_failures} 次)")
                        if self._consecutive_failures >= 2:
                            # 连续失败两次，强制让 LLM 基于自身知识给出 Final Answer
                            self._consecutive_failures = 0
                            force_messages = list(current_messages)
                            force_messages[-1]["content"] = (
                                react_context +
                                f"Observation: {observation}\n"
                                "Thought: 工具连续调用失败，无法检索教材，我将根据自身知识给出解答并提示学生。\n"
                                "Final Answer: "
                            )
                            final_text = self.llm_client.call(
                                force_messages,
                                temperature=0.5,
                                max_tokens=max_tokens,
                                model=model,
                            ) or "抱歉，当前教材检索服务暂时不可用（API 额度已用尽），请稍后再试。"
                            final_text = re.sub(r'^Final Answer:\s*', '', final_text.strip())
                            logger.info("[Tutor Agent] 工具连续失败后强制生成最终回答。")
                            final_text = self._humanize_answer(
                                query=query,
                                conversation_history=conversation_history,
                                text_results=accumulated_text_results,
                                image_hits=accumulated_image_hits,
                                draft_answer=final_text,
                                model=model,
                                max_tokens=max_tokens,
                            )
                            return final_text, accumulated_image_hits, accumulated_text_results
                    else:
                        self._consecutive_failures = 0

                    react_context += f"Observation: {observation}\n"
                    continue
                else:
                    logger.info("[Tutor Agent] 模型决定无需调用工具 (Action: None)，直接回答。")
                    react_context += response_text + "\nObservation: 没有工具被调用。\n"
            else:
                react_context += response_text + "\n"

            # 提取 Final Answer
            final_answer_match = re.search(r"Final Answer:\s*(.*)", response_text, re.DOTALL)
            if final_answer_match:
                final_answer = final_answer_match.group(1).strip()
                draft_final_answer = final_answer
                if not has_tool_call:
                    # 课程相关问题默认必须先检索，避免模型臆断。
                    react_context += (
                        "Observation: 你尚未调用任何工具。"
                        "对课程事实类问题必须先检索教材，请先调用 search_textbook_text。\n"
                    )
                    logger.warning("[Tutor Agent] 检测到未检索即回答，已要求模型先检索再作答。")
                    continue
                logger.info("[Tutor Agent] 思考完毕，生成并输出最终回答。")
                final_answer = self._humanize_answer(
                    query=query,
                    conversation_history=conversation_history,
                    text_results=accumulated_text_results,
                    image_hits=accumulated_image_hits,
                    draft_answer=final_answer,
                    model=model,
                    max_tokens=max_tokens,
                )
                return final_answer, accumulated_image_hits, accumulated_text_results
            else:
                logger.warning(f"[Tutor Agent] 输出格式不符合要求，将其作为 Final Answer 处理:\n{response_text}")
                if not has_tool_call:
                    react_context += (
                        "Observation: 你的输出缺少规范格式，且尚未调用工具。"
                        "请先调用 search_textbook_text 再回答。\n"
                    )
                    continue
                # LLM 直接输出了一段话，没有按格式走 — 视为最终回答
                draft_final_answer = response_text.strip()
                final_answer = self._humanize_answer(
                    query=query,
                    conversation_history=conversation_history,
                    text_results=accumulated_text_results,
                    image_hits=accumulated_image_hits,
                    draft_answer=draft_final_answer,
                    model=model,
                    max_tokens=max_tokens,
                )
                return final_answer, accumulated_image_hits, accumulated_text_results

        logger.warning(f"[Tutor Agent] 达到最大迭代次数 ({self.max_iterations})")
        fallback = draft_final_answer or "这个问题我思考了很久也没有得出结论，能换个方式问我吗？"
        fallback = self._humanize_answer(
            query=query,
            conversation_history=conversation_history,
            text_results=accumulated_text_results,
            image_hits=accumulated_image_hits,
            draft_answer=fallback,
            model=model,
            max_tokens=max_tokens,
        )
        return fallback, accumulated_image_hits, accumulated_text_results
