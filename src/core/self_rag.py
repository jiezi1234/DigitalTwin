"""
Self-RAG 服务模块
通过内联反思标签（[Retrieve] / [IsRel] / [IsSup] / [IsUse]）让系统自主判断是否检索、
评估检索质量、验证生成结果。支持 HuggingFace selfrag 模型和通义千问 prompt 模拟两种后端。

新流程（2-3 次 API 调用）：
  Call 1: [Retrieve] 检索决策
  Stage 2: 向量检索
  Call 2: [IsRel] + 生成 + [IsSup] + [IsUse]（合并为一次调用）
  [Call 3: 条件重试]
"""

import os
import re
import uuid
import logging
import requests
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ── 模块级正则常量 ────────────────────────────────────────────
RE_RETRIEVE  = re.compile(r'\[Retrieve\]\s*(是|否)')
RE_ISREL     = re.compile(r'\[IsRel\]\s*(\d+)\s*(相关|不相关)')
RE_ISSUP     = re.compile(r'\[IsSup\]\s*(完全支持|部分支持|无支持)')
RE_ISUSE     = re.compile(r'\[IsUse\]\s*(\d)')
RE_ANY_TOKEN = re.compile(r'\[(?:Retrieve|IsRel|IsSup|IsUse)\][^\n]*(?:\n|$)')


def _trunc(text: str, limit: int = 300) -> str:
    """截断文本用于日志输出"""
    if len(text) <= limit:
        return text
    return text[:limit] + f"...({len(text)}字)"


class SelfRAGService:
    """Self-RAG 反思增强检索生成服务"""

    def __init__(
        self,
        backend: str = "auto",
        critique_enabled: bool = True,
        # Qwen 后端配置
        dashscope_api_key: str = "",
        llm_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode",
        chat_model: str = "qwen-plus",
        # 生成参数（避免从 app.Config 延迟导入）
        temperature: float = 0.5,
        top_p: float = 0.7,
        repetition_penalty: float = 1.2,
        qwen_api_path: str = "/v1/chat/completions",
        rag_system_prefix: str = "",
        rag_role_instruction: str = "",
        # HF 后端配置
        hf_model_name: str = "selfrag/selfrag_llama2_7b",
        hf_device: str = "auto",
        # 阈值
        relevance_threshold: float = 0.5,
        support_threshold: float = 0.5,
        utility_threshold: int = 3,
    ):
        self.backend = backend  # "auto" | "hf" | "qwen"
        self.critique_enabled = critique_enabled

        # Qwen
        self.api_key = dashscope_api_key
        self.llm_api_base = llm_api_base.rstrip("/")
        self.chat_model = chat_model

        # 生成参数
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.qwen_api_path = qwen_api_path
        self.rag_system_prefix = rag_system_prefix
        self.rag_role_instruction = rag_role_instruction

        # HF — 懒加载，首次调用时才下载/加载模型
        self.hf_model_name = hf_model_name
        self.hf_device = hf_device
        self._hf_tokenizer = None
        self._hf_model = None

        # 阈值
        self.relevance_threshold = relevance_threshold
        self.support_threshold = support_threshold
        self.utility_threshold = utility_threshold

        # 运行时状态：auto 模式下 HF 失败后锁定 qwen
        self._hf_available: Optional[bool] = None  # None=未测试

        logger.info(
            "SelfRAGService 初始化完成 (backend=%s, critique=%s)",
            self.backend, self.critique_enabled,
        )

    # ── 主入口 ──────────────────────────────────────────────

    def run(
        self,
        query: str,
        rag_service,
        persona: Optional[dict],
        conversation: List[Dict[str, str]],
        system_prompt: str,
        max_tokens: int = 500,
        rag_config: Optional[dict] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        编排完整 Self-RAG 流程。

        Returns:
            (reply, error) — reply 为生成的回复文本，error 为错误信息（二者互斥）。
        """
        req_id = uuid.uuid4().hex[:8]
        logger.info("[req=%s] 用户输入: %s", req_id, query)

        try:
            # 1. [Retrieve] 检索决策
            need_retrieval = self._stage_retrieve_decision(query, req_id=req_id)

            if not need_retrieval:
                reply, is_use = self._generate_no_retrieval(
                    query, persona, conversation, system_prompt, max_tokens, req_id=req_id
                )
                return reply, None

            # 2. 向量检索
            if not rag_service:
                logger.warning("[req=%s] RAGService 不可用，回退到无检索生成", req_id)
                reply, is_use = self._generate_no_retrieval(
                    query, persona, conversation, system_prompt, max_tokens, req_id=req_id
                )
                return reply, None

            rp = (persona or {}).get("rag_params", {})
            cfg = rag_config or {}
            results = rag_service.search(
                query=query,
                persona=persona,
                k=rp.get("k", cfg.get("max_results", 20)),
                include_nearby=rp.get("include_nearby", True),
                time_window_minutes=rp.get("time_window_minutes", 30),
                nearby_per_result=rp.get("nearby_per_result", 8),
                max_total_results=rp.get("max_total_results", 50),
                lambda_mult=rp.get("lambda_mult", 0.6),
            )

            if not results:
                logger.info("[req=%s] 向量检索: 无结果，回退到无检索生成", req_id)
                reply, is_use = self._generate_no_retrieval(
                    query, persona, conversation, system_prompt, max_tokens, req_id=req_id
                )
                return reply, None

            # 记录向量检索结果
            search_lines = []
            for i, (content, meta, score) in enumerate(results[:20], 1):
                search_lines.append(f"  [{i}] score={score:.3f} {content.strip().replace(chr(10), ' ')[:80]}")
            logger.info("[req=%s] 向量检索: %d 条结果\n%s", req_id, len(results), "\n".join(search_lines))

            # 3. 合并调用: [IsRel] + 生成 + [IsSup] + [IsUse]
            reply, is_rel, is_sup, is_use = self._combined_generate(
                query, results, rag_service, persona, conversation,
                system_prompt, max_tokens, cfg, req_id=req_id,
            )

            if not reply:
                return None, "Self-RAG 生成阶段失败"

            # 4. 质量检查 & 条件重试
            if self.critique_enabled:
                need_retry = False
                if is_sup == "无支持":
                    need_retry = True
                elif is_sup == "部分支持" and is_use < self.utility_threshold:
                    need_retry = True
                elif is_use < self.utility_threshold:
                    need_retry = True

                if need_retry:
                    logger.info(
                        "[req=%s] 评估未通过 (IsSup=%s, IsUse=%d, 阈值=%d)，触发重试",
                        req_id, is_sup, is_use, self.utility_threshold,
                    )
                    retry_reply, _, _, _ = self._combined_generate(
                        query, results, rag_service, persona, conversation,
                        system_prompt, max_tokens, cfg, req_id=req_id,
                    )
                    if retry_reply:
                        reply = retry_reply

            return reply, None

        except Exception as e:
            logger.error("[req=%s] Self-RAG 流程异常: %s", req_id, e, exc_info=True)
            return None, f"Self-RAG 流程异常: {e}"

    # ── 反思调度器（纯路由，不记日志）─────────────────────────

    def _reflect(self, prompt: str, task: str, req_id: str = "") -> Optional[str]:
        """统一调度器，按配置选择 HF 或 Qwen 后端执行反思任务。"""
        backend = self._resolve_backend()

        if backend == "hf":
            result = self._reflect_via_hf(prompt, req_id=req_id)
            if result is not None:
                return result
            if self.backend == "auto":
                logger.warning("[req=%s] HF 后端失败，回退到 Qwen", req_id)
                self._hf_available = False
                return self._reflect_via_qwen(prompt, task, req_id=req_id)
            return None

        return self._reflect_via_qwen(prompt, task, req_id=req_id)

    def _resolve_backend(self) -> str:
        """解析实际使用的后端。"""
        if self.backend == "qwen":
            return "qwen"
        if self.backend == "hf":
            return "hf"
        # auto
        if self._hf_available is False:
            return "qwen"
        return "hf"

    # ── HF 后端（本地 transformers 推理）──────────────────────

    def _load_hf_model(self) -> bool:
        """懒加载 HF selfrag 模型，首次调用时下载并加载到显存/内存。"""
        if self._hf_model is not None:
            return True
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            logger.info("正在加载 HF 模型: %s ...", self.hf_model_name)

            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                self.hf_model_name, padding_side="left"
            )
            if self._hf_tokenizer.pad_token is None:
                self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token

            if self.hf_device == "auto":
                device_map = "auto" if torch.cuda.is_available() else "cpu"
            else:
                device_map = self.hf_device

            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device_map,
            )

            logger.info("HF 模型加载完成 (device_map=%s)", device_map)
            return True

        except ImportError:
            logger.warning("transformers 或 torch 未安装，HF 后端不可用")
            return False
        except Exception as e:
            logger.warning("HF 模型加载失败: %s", e)
            return False

    def _reflect_via_hf(self, prompt: str, req_id: str = "") -> Optional[str]:
        """使用本地 transformers 模型进行推理（只记录错误）。"""
        if not self._load_hf_model():
            self._hf_available = False
            return None

        try:
            inputs = self._hf_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self._hf_model.device) for k, v in inputs.items()}

            import torch
            with torch.no_grad():
                outputs = self._hf_model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                )

            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            raw_text = self._hf_tokenizer.decode(new_tokens, skip_special_tokens=False)

            self._hf_available = True
            return self._clean_reflection_tokens(raw_text)

        except Exception as e:
            logger.warning("[req=%s] HF 本地推理异常: %s", req_id, e)
            self._hf_available = False
            return None

    # ── Qwen 后端（只记录错误）────────────────────────────────

    def _reflect_via_qwen(self, prompt: str, task: str, req_id: str = "") -> Optional[str]:
        """调用 DashScope OpenAI 兼容端点（反思专用，轻量级）。"""
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 300,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        endpoint = f"{self.llm_api_base}/v1/chat/completions"

        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
            else:
                logger.warning("[req=%s] Qwen 反思 API 返回 %d: %s", req_id, resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("[req=%s] Qwen 反思 API 异常: %s", req_id, e)

        return None

    # ── Stage 1: [Retrieve] 决策 ─────────────────────────────

    def _stage_retrieve_decision(self, query: str, req_id: str = "") -> bool:
        """判断是否需要从聊天记录中检索，解析 [Retrieve] 标签。"""
        prompt = (
            "你是一个检索决策助手。判断以下用户消息是否需要从聊天记录数据库中检索信息来回答。\n\n"
            "不需要检索的情况：简单问候、询问身份、对话延续、通用知识问题。\n"
            "需要检索的情况：询问某人说过什么、特定事件或话题、涉及聊天记录中的人物或事件。\n\n"
            f"用户消息：{query}\n\n"
            "请用以下格式回答（只输出这一行，不要解释）：\n"
            "[Retrieve] 是\n"
            "或\n"
            "[Retrieve] 否"
        )
        result = self._reflect(prompt, "retrieve_decision", req_id=req_id)

        if result is None:
            logger.info("[req=%s] [Retrieve] API返回: 失败 → 默认需要检索", req_id)
            return True

        m = RE_RETRIEVE.search(result)
        if m:
            need = m.group(1) == "是"
        else:
            need = "是" in result

        logger.info("[req=%s] [Retrieve] API返回: '%s' → %s", req_id, result.strip(), "需要检索" if need else "无需检索")
        return need

    # ── Call 2: 合并生成 ─────────────────────────────────────

    def _combined_generate(
        self,
        query: str,
        results: List[Tuple[str, Dict[str, Any], float]],
        rag_service,
        persona: Optional[dict],
        conversation: List[Dict[str, str]],
        system_prompt: str,
        max_tokens: int,
        cfg: dict,
        req_id: str = "",
    ) -> Tuple[Optional[str], Dict[int, bool], str, int]:
        """
        分两步进行：
        Call 2a: 判断段落相关性 [IsRel]
        过滤后
        Call 2b: 基于相关段落生成回复 + [IsSup] + [IsUse]

        Returns:
            (reply, is_rel_map, is_sup, is_use)
        """
        # --- Call 2a: 相关性评估 ---
        passages_text = self._build_numbered_passages(results)

        isrel_instruction = (
            "你是一个信息相关性判断助手，不需要扮演任何角色。\n"
            "判断以下检索到的历史聊天记录段落，逐一判断它们是否与用户最后提出的问题相关。\n\n"
            f"检索段落：\n{passages_text}\n\n"
            "判断标准：段落包含能部分或完全回答用户问题的信息=相关；内容与问题完全无关=不相关。\n"
            "【重要】请严格按照下面格式逐行输出，只输出这些行，不要解释也不要其他文字：\n"
            "[IsRel] 1 相关\n"
            "[IsRel] 2 不相关\n"
            "[IsRel] 3 相关\n"
            "...（对每条段落各输出一行）"
        )

        # Call 2a 单独使用中立 system prompt，不带分身扮演指令，避免角色指令干扰格式输出
        isrel_messages = [
            {"role": "system", "content": isrel_instruction},
            # 只放最后一条用户消息，让模型明确知道在评估什么问题
            {"role": "user", "content": conversation[-1]["content"] if conversation else query},
        ]

        logger.info("[req=%s] [Call 2a IsRel] 请求API评估 %d 条段落...", req_id, min(len(results), 15))
        isrel_raw = self._call_qwen_api(isrel_messages, 400, req_id=req_id)
        
        is_rel_map = {}
        if isrel_raw:
            # 完整打印 API 原始返回，不截断，方便确认模型真的在做判断
            logger.info("[req=%s] [Call 2a IsRel] API原始返回 (%d字):\n%s", req_id, len(isrel_raw), isrel_raw)
            is_rel_map = self._parse_isrel_tokens(isrel_raw, len(results[:15]))
            if not is_rel_map:
                logger.warning("[req=%s] [Call 2a IsRel] 标签解析失败（格式不符），默认全部相关", req_id)
                is_rel_map = {i: True for i in range(1, len(results[:15]) + 1)}
        else:
            logger.warning("[req=%s] [Call 2a IsRel] API无返回，默认全部相关", req_id)
            is_rel_map = {i: True for i in range(1, len(results[:15]) + 1)}

        # 逐条打印段落内容 + [IsRel] 标签
        detail_lines = []
        for i, (content, meta, score) in enumerate(results[:15], 1):
            tag = "[IsRel] 相关" if is_rel_map.get(i, True) else "[IsRel] 不相关"
            source = meta.get('_result_source', 'semantic')  # semantic=向量检索 temporal=时间邻近
            preview = content.strip().replace("\n", " ")[:100]
            detail_lines.append(f"  [{i}] {tag}  来源={source}  {preview}")
        logger.info("[req=%s] [Call 2a IsRel] 各段落判断：\n%s", req_id, "\n".join(detail_lines))

        # --- 过滤相关段落 ---
        relevant_results = []
        for i, item in enumerate(results[:15], 1):
            if is_rel_map.get(i, True): # 没评出的默认相关
                relevant_results.append(item)
        
        # 补上未评估的后备结果
        if len(results) > 15:
            relevant_results.extend(results[15:])
            
        logger.info("[req=%s] [过滤] 保留相关段落: %d -> %d 条", req_id, len(results), len(relevant_results))

        # --- Call 2b: 生成 + 评估 ---
        if not relevant_results:
            logger.info("[req=%s] [Call 2b] 无相关段落，转向无检索生成", req_id)
            # 无相关段落，直接走无检索生成
            reply, is_use = self._generate_no_retrieval(
                query, persona, conversation, system_prompt, max_tokens, req_id=req_id
            )
            return reply, is_rel_map, "无支持", is_use

        # 格式化 RAG 上下文（仅用相关段落）
        context = rag_service.format_context(
            relevant_results,
            max_context_length=cfg.get("max_context_length", 2000),
            include_metadata=cfg.get("include_metadata", True),
        )

        generate_instruction = (
            "\n\n【Self-RAG 生成与反思指令】\n"
            "第一步：基于上方的历史聊天记录，以角色身份自然回复用户。回复内容本身不要包含任何方括号标记。\n\n"
            "第二步：在回复完全结束后，换行输出以下两个自我评估标记：\n"
            "[IsSup] 完全支持/部分支持/无支持\n"
            "（评估你的回复是否被上方的记录事实所支持)\n"
            "[IsUse] 1-5\n"
            "（评估你的回复对用户问题的有用程度，5分最有用）\n\n"
            "请严格输出这两个标记并放在最后。"
        )

        rag_text = self.rag_system_prefix + "\n" + context
        combined_system = f"{rag_text}\n\n{system_prompt}{self.rag_role_instruction}{generate_instruction}"

        gen_messages = [{"role": "system", "content": combined_system}]
        gen_messages.extend(conversation)

        logger.info("[req=%s] [Call 2b 生成] 基于 %d 条相关段落发送生成请求...", req_id, len(relevant_results))
        gen_raw = self._call_qwen_api(gen_messages, max_tokens + 200, req_id=req_id)
        
        if not gen_raw:
            return None, is_rel_map, "部分支持", 3

        logger.info("[req=%s] [Call 2b 生成] API返回:\n%s", req_id, _trunc(gen_raw, 500))

        is_sup = self._parse_issup_token(gen_raw)
        is_use = self._parse_isuse_token(gen_raw)
        
        # 提取去掉标记的纯回复
        raw_lines = gen_raw.split('\n')
        reply_lines = []
        for line in raw_lines:
            if RE_ISSUP.search(line) or RE_ISUSE.search(line) or RE_ISREL.search(line):
                continue
            reply_lines.append(line)
            
        reply = "\n".join(reply_lines).strip()
        reply = self._strip_all_tokens(reply).strip() or None

        logger.info(
            "[req=%s] [Call 2b 解析] IsSup=%s IsUse=%d 回复(%d字)",
            req_id, is_sup, is_use, len(reply) if reply else 0,
        )

        return reply, is_rel_map, is_sup, is_use

    def _generate_no_retrieval(
        self,
        query: str,
        persona: Optional[dict],
        conversation: List[Dict[str, str]],
        system_prompt: str,
        max_tokens: int,
        req_id: str = "",
    ) -> Tuple[Optional[str], int]:
        """
        无检索路径生成回复，回复末尾带 [IsUse] 标签。

        Returns:
            (reply, is_use)
        """
        selfrag_instruction = (
            "\n\n【Self-RAG 反思指令】\n"
            "以角色身份自然回复用户后，在回复末尾换行输出：\n"
            "[IsUse] 1-5\n"
            "评估你的回复对用户问题的有用程度（5分最有用）。"
        )

        combined_system = f"{system_prompt}{self.rag_role_instruction}{selfrag_instruction}"

        messages = [{"role": "system", "content": combined_system}]
        messages.extend(conversation)

        raw = self._call_qwen_api(messages, max_tokens + 200, req_id=req_id)
        if raw is None:
            return None, 3

        is_use = self._parse_isuse_token(raw)
        reply = self._strip_all_tokens(raw).strip()

        logger.info("[req=%s] [无检索生成] API返回: %s → IsUse=%d 回复(%d字)", req_id, _trunc(raw, 300), is_use, len(reply) if reply else 0)

        return reply if reply else None, is_use

    # ── Qwen API 公共调用（只记录错误）────────────────────────

    def _call_qwen_api(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        req_id: str = "",
    ) -> Optional[str]:
        """调用 Qwen 生成 API（DRY 公共方法）。"""
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        endpoint = f"{self.llm_api_base}/{self.qwen_api_path.lstrip('/')}"

        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
            else:
                logger.warning("[req=%s] Qwen 生成 API 返回 %d: %s", req_id, resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("[req=%s] Qwen 生成 API 异常: %s", req_id, e)

        return None

    # ── 段落构建 ─────────────────────────────────────────────

    @staticmethod
    def _build_numbered_passages(
        results: List[Tuple[str, Dict[str, Any], float]],
        max_passages: int = 15,
        max_preview: int = 120,
    ) -> str:
        """构建编号段落列表，每条截断 max_preview 字。"""
        lines = []
        for i, (content, meta, _score) in enumerate(results[:max_passages], 1):
            text = content.strip().replace("\n", " ")[:max_preview]
            lines.append(f"[{i}] {text}")
        return "\n".join(lines)

    # ── 标签解析方法（纯解析，不记日志）───────────────────────

    @staticmethod
    def _parse_isrel_tokens(text: str, passage_count: int) -> Dict[int, bool]:
        """
        解析 [IsRel] 标签。支持多种 Qwen 可能输出的格式：
          标准格式:  [IsRel] 1 相关
          列表格式:  1. 相关  /  1: 不相关  /  1) 相关
          文字格式:  段落1：相关
        返回空字典时表示完全解析失败（由调用方决定如何处理）。
        """
        found: Dict[int, bool] = {}

        # 优先：标准 [IsRel] N 相关/不相关
        for m in RE_ISREL.finditer(text):
            idx = int(m.group(1))
            found[idx] = (m.group(2) == "相关")

        if found:
            return found

        # 备用：纯数字列表格式，如 "1. 相关"、"2: 不相关"、"3) 相关"
        fallback = re.compile(
            r'(?:^|\n)\s*(\d+)[\.:\)]\s*(相关|不相关)',
            re.MULTILINE,
        )
        for m in fallback.finditer(text):
            idx = int(m.group(1))
            if 1 <= idx <= passage_count:
                found[idx] = (m.group(2) == "相关")

        if found:
            return found

        # 备用：段落N：相关/不相关
        para_pat = re.compile(r'段落\s*(\d+)\s*[：:]\s*(相关|不相关)')
        for m in para_pat.finditer(text):
            idx = int(m.group(1))
            if 1 <= idx <= passage_count:
                found[idx] = (m.group(2) == "相关")

        return found  # 可能仍为 {}，由调用方判断

    @staticmethod
    def _parse_issup_token(text: str) -> str:
        """解析 [IsSup] 标签，缺失时默认 '部分支持'。"""
        m = RE_ISSUP.search(text)
        return m.group(1) if m else "部分支持"

    @staticmethod
    def _parse_isuse_token(text: str) -> int:
        """解析 [IsUse] 标签，缺失时默认 3。"""
        m = RE_ISUSE.search(text)
        return max(1, min(5, int(m.group(1)))) if m else 3

    @staticmethod
    def _extract_reply_text(text: str) -> Optional[str]:
        """
        从合并生成的输出中提取纯回复文本。
        策略：最后一个 [IsRel] 行之后、第一个 [IsSup]/[IsUse] 行之前。
        """
        lines = text.split("\n")

        last_isrel_idx = -1
        for i, line in enumerate(lines):
            if RE_ISREL.search(line):
                last_isrel_idx = i

        first_eval_idx = len(lines)
        for i in range(last_isrel_idx + 1, len(lines)):
            if RE_ISSUP.search(lines[i]) or RE_ISUSE.search(lines[i]):
                first_eval_idx = i
                break

        if last_isrel_idx >= 0:
            reply_lines = lines[last_isrel_idx + 1:first_eval_idx]
        else:
            reply_lines = lines[:first_eval_idx]

        reply = "\n".join(reply_lines).strip()
        if reply:
            return SelfRAGService._strip_all_tokens(reply).strip() or None

        fallback = SelfRAGService._strip_all_tokens(text).strip()
        return fallback if fallback else None

    @staticmethod
    def _strip_all_tokens(text: str) -> str:
        """清除所有 Self-RAG 反思标签，返回干净文本。"""
        return RE_ANY_TOKEN.sub("", text)

    # ── 工具方法 ─────────────────────────────────────────────

    @staticmethod
    def _clean_reflection_tokens(text: str) -> str:
        """清洗 HF 模型输出中的反思标记（<paragraph>、[Retrieval] 等）。"""
        text = re.sub(r"<paragraph>.*?</paragraph>", "", text, flags=re.DOTALL)
        text = re.sub(r"\[.*?\]", "", text)
        text = text.replace("<s>", "").replace("</s>", "")
        text = re.sub(r"\s+", " ", text).strip()
        return text
