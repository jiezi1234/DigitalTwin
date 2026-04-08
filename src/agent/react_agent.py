import re
import logging
from typing import List, Dict, Any, Tuple

from src.infrastructure.llm_client import LLMClient
from src.skills.base_skill.scripts.base_skill import BaseSkill

logger = logging.getLogger(__name__)

# ReAct 提示词模板
REACT_SYSTEM_PROMPT = """你是一个具备反思和工具调用能力的 AI 分身。

{persona_system_prompt}

你可以使用以下技能（工具）：
{tools_desc}

请遵循以下 ReAct 思维模式来回答用户的问题，每一步独立占一行。你的输出必须严格遵循以下格式（中间不要加空行）：

Thought: 你在回答问题前需要思考什么（如果不需要查找资料直接回答即可）
Action: 要使用的工具名称（仅限 {tool_names}，如果可以直接回答，请返回 None）
Action Input: 传入工具的参数
Observation: 工具的返回结果（此步骤由系统填充，你自己不要输出Observation）
... (思考/行动/观察 循环最多进行几次)
Thought: 我现在知道了最终答案
Final Answer: 你作为该数字分身的最终回复（风格必须贴合预设要求）

注意：
1. 你的职责不仅是解决任务，还需要像设定里的人设一样回答。
2. 只有在你确实不记得或者不知道细节时，才调用 'retrieve_chat_history'。如果只是纯寒暄、打招呼，请产生 Thought 后，将 Action 置为 None。
3. 当你决定要输出最终答案时，只需输出 Thought，和 Final Answer。
4. 绝对严禁你编造 Observation！当你输出完 'Action Input: ...' 之后，必须立即停笔，绝对不要自己在同一回合里输出 'Observation:' 或 'Final Answer:'，等待系统给你返回结果！

示例 1 (需要检索的场景):
Thought: 用户问我昨天吃了什么，我记不清了，需要从聊天记录查询。
Action: retrieve_chat_history
Action Input: 昨天 吃 晚饭
(此时立即停止输出，等待系统返回 Observation)

示例 2 (无需检索，直接寒暄):
Thought: 用户在向我问好，不需要检索记录。
Action: None
Action Input: None
Thought: 我现在可以直接回答了。
Final Answer: 哈喽！今天怎么样？

开始！
"""

class ReActAgent:
    """基于 ReAct (Reasoning and Acting) 模式的智能体"""

    def __init__(
        self,
        llm_client: LLMClient,
        tools: List[BaseSkill],
        max_iterations: int = 5
    ):
        self.llm_client = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations

    def run(
        self, 
        query: str, 
        conversation_history: List[Dict[str, str]], 
        persona: Dict[str, Any],
        max_tokens: int = 800
    ) -> Tuple[str, Dict[str, Any]]:
        """
        运行 ReAct 主循环
        """
        tools_desc = "\n".join([f"- {name}: {t.description}" for name, t in self.tools.items()])
        tool_names = ", ".join(self.tools.keys())
        
        persona_sys_prompt = persona.get("system_prompt", "")
        system_prompt = REACT_SYSTEM_PROMPT.format(
            persona_system_prompt=persona_sys_prompt,
            tools_desc=tools_desc,
            tool_names=tool_names
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history[-10:])
        
        react_context = f"Question: {query}\n"
        
        eval_stats = {
            "iterations": 0,
            "actions_taken": [],
            "retrieved": False
        }

        # 主循环
        for i in range(self.max_iterations):
            eval_stats["iterations"] = i + 1
            
            current_messages = list(messages)
            current_messages.append({"role": "user", "content": react_context})

            response_text = self.llm_client.call(
                current_messages, 
                temperature=0.3,
                max_tokens=max_tokens
            )

            if not response_text:
                logger.error("[Agent Workflow] LLM 返回空。")
                return "抱歉，我现在有些卡顿，无法思考和作答。", eval_stats

            # 打印完整的 LLM Thought 流
            logger.info(f"========== [Agent Workflow] 迭代回合 {i+1} ==========\n模型内部思考与决策:\n{response_text}\n==================================================")

            # 优先提取 Action & Action Input
            action_match = re.search(r"Action:\s*(.*?)\n", response_text + "\n")
            action_input_match = re.search(r"Action Input:\s*(.*?)\n", response_text + "\n")

            if action_match and action_input_match:
                action = action_match.group(1).strip()
                action_input = action_input_match.group(1).strip()
                
                # 如果要真正调用行动
                if action and action != "None" and action.lower() != "none":
                    # 消除由于大模型幻觉随手乱编的 Observation 甚至 Final Answer，严格切割到 Action Input
                    clean_thought_match = re.search(r"(.*?Action Input:.*?\n)", response_text + "\n", re.DOTALL)
                    react_context += (clean_thought_match.group(1) if clean_thought_match else response_text)
                    
                    if action not in self.tools:
                        logger.warning(f"[Agent Workflow] 未知技能: {action}")
                        react_context += f"Observation: 技能 {action} 不存在，请使用: {tool_names}\n"
                        continue

                    logger.info(f"\n>>>> [Agent Workflow] 开始调用外部技能 <<<<\n技能名称: {action}\n传入参数: {action_input}")
                    tool = self.tools[action]
                    
                    observation = tool.run(action_input)
                    
                    eval_stats["actions_taken"].append(action)
                    if action == "retrieve_chat_history":
                        eval_stats["retrieved"] = True
                    
                    logger.info(f"<<<< [Agent Workflow] 技能调用结束，返回的完整数据结果如下 >>>>\n{observation}\n" + "-"*50)
                    react_context += f"Observation: {observation}\n"
                    continue
                else:
                    logger.info("[Agent Workflow] 模型决定无需调用外部技能 (Action: None)。下一步将直接回答。")
                    react_context += response_text + "\nObservation: 没有任何工具被调用。\n"
            else:
                react_context += response_text + "\n"

            # 只有没有调用外部工具时才能寻找 Final Answer
            final_answer_match = re.search(r"Final Answer:\s*(.*)", response_text, re.DOTALL)
            if final_answer_match:
                final_answer = final_answer_match.group(1).strip()
                logger.info(f"[Agent Workflow] 思考完毕，生成并输出最终回答。")
                return final_answer, eval_stats
            else:
                logger.warning(f"[Agent Workflow] 无效的输出格式:\n {response_text}")
                react_context += "Observation: 你的输出不符合要求的格式。请一定要包含 Action/Action Input 或 Final Answer。\n"
        
        logger.warning(f"[Agent Workflow] 达到最大内部迭代次数 ({self.max_iterations})")
        return "不好意思，这个问题我思考了很久也没得出结论，能换个方式问我吗？", eval_stats
