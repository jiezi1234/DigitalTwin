import logging
from typing import List, Dict, Any, Optional, Tuple
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.db_client import DBClient
from src.rag.rag_engine import RAGEngine
from src.rag.query_processor import QueryProcessor
from src.rag.self_rag import SelfRAG
from src.infrastructure.telemetry import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class RAGService:
    """分身 RAG 服务 (集成架构)"""

    def __init__(
        self,
        llm_client: LLMClient,
        db_client: DBClient,
        collection_name: str = "wechat_embeddings",
        enable_coreference_resolution: bool = True,
        enable_query_rewriting: bool = True,
        enable_self_rag: bool = False,
        self_rag_mode: str = "chat",
    ):
        """
        初始化 RAG 服务

        Args:
            llm_client: LLM 客户端
            db_client: 数据库客户端
            collection_name: 向量集合名称
            enable_coreference_resolution: 启用指代消解
            enable_query_rewriting: 启用 Query Rewriting
            enable_self_rag: 启用 Self-RAG 反思逻辑
            self_rag_mode: Self-RAG 检索决策模式 ("chat" 聊天记录 / "knowledge" 知识库)
        """
        self.llm_client = llm_client
        self.db_client = db_client
        self.collection_name = collection_name

        # 初始化核心组件
        self.rag_engine = RAGEngine(db_client=db_client)
        self.query_processor = QueryProcessor(
            llm_client=llm_client,
            enable_coreference_resolution=enable_coreference_resolution,
            enable_query_rewriting=enable_query_rewriting,
        )

        # 根据模式选择 Self-RAG 检索决策提示词
        self_rag_prompts = {}
        if enable_self_rag and self_rag_mode == "knowledge":
            self_rag_prompts["decide_retrieval"] = SelfRAG.KNOWLEDGE_DECIDE_PROMPT
        self.self_rag = SelfRAG(llm_client=llm_client, prompts=self_rag_prompts) if enable_self_rag else None

    def search(
        self,
        query: str,
        persona: Optional[Dict[str, Any]] = None,
        k: int = 15,
        lambda_mult: float = 0.6,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """搜索相关聊天记录"""
        return self.rag_engine.search(
            query=query,
            collection_name=self.collection_name,
            query_processor=self.query_processor,
            k=k,
            lambda_mult=lambda_mult,
            persona=persona,
        )

    def format_context(
        self,
        results: List[Tuple[str, Dict[str, Any], float]],
        max_context_length: int = 2000,
        include_metadata: bool = True,
    ) -> str:
        """格式化搜索结果"""
        return self.rag_engine.format_context(
            results,
            max_context_length=max_context_length,
            include_metadata=include_metadata,
            format_type="chat",
        )

    def chat(
        self,
        query: str,
        conversation: List[Dict[str, str]],
        persona: Dict[str, Any],
        system_prefix: str = "相关记录：\n",
        role_instruction: str = "",
        max_tokens: int = 500
    ) -> Tuple[str, Dict[str, Any]]:
        """
        全自动化 RAG 对话逻辑
        
        Returns:
            (回复文本, 评估元数据)
        """
        with tracer.start_as_current_span("rag_service.chat") as span:
            logger.debug(f"[用户输入] {query}")
            eval_stats = {"retrieved": False}

            # 1. Self-RAG: 判定是否需要检索
            need_retrieval = True
            if self.self_rag:
                need_retrieval = self.self_rag.decide_retrieval(query)
                eval_stats["retrieve_decision"] = need_retrieval

            results = []
            context_text = ""

            if need_retrieval:
                # 2. 执行检索
                results = self.search(query, persona=persona)
                eval_stats["retrieved"] = True
                eval_stats["raw_count"] = len(results)

                if results and self.self_rag:
                    # 3. Self-RAG: 评估相关性
                    is_rel_map = self.self_rag.evaluate_relevance(query, results)
                    eval_stats["relevance_map"] = is_rel_map
                    # 过滤掉不相关的段落
                    results = [res for i, res in enumerate(results, 1) if is_rel_map.get(i, True)]
                    eval_stats["filtered_count"] = len(results)

                if results:
                    context_text = self.format_context(results)

            # 4. 构建提示词并调用生成
            reply, is_sup, is_use = self._generate(
                query, conversation, persona, context_text,
                system_prefix, role_instruction, max_tokens,
            )

            if not reply:
                return "抱歉，我目前无法回答这个问题。", eval_stats

            if self.self_rag:
                eval_stats["is_sup"] = is_sup
                eval_stats["is_use"] = is_use

                # 5. 质量检查 & 条件重试
                utility_threshold = self.self_rag.thresholds.get("utility", 3)
                need_retry = (
                    is_sup == "无支持"
                    or (is_sup == "部分支持" and is_use < utility_threshold)
                    or is_use < utility_threshold
                )
                if need_retry:
                    logger.info(
                        f"[Self-RAG 重试] 评估未通过 (IsSup={is_sup}, IsUse={is_use}, "
                        f"阈值={utility_threshold})，触发重试"
                    )
                    eval_stats["retried"] = True
                    retry_reply, retry_sup, retry_use = self._generate(
                        query, conversation, persona, context_text,
                        system_prefix, role_instruction, max_tokens,
                    )
                    if retry_reply:
                        reply = retry_reply
                        eval_stats["is_sup"] = retry_sup
                        eval_stats["is_use"] = retry_use

            logger.debug(f"[最终输出] {reply}")
            return reply, eval_stats

    def _generate(
        self,
        query: str,
        conversation: List[Dict[str, str]],
        persona: Dict[str, Any],
        context_text: str,
        system_prefix: str,
        role_instruction: str,
        max_tokens: int,
    ) -> Tuple[str, str, int]:
        """
        构建 prompt 并调用 LLM 生成回复。

        Returns:
            (回复文本, is_sup, is_use)
        """
        full_system = f"{persona['system_prompt']}\n\n{role_instruction}"
        if context_text:
            full_system = f"{system_prefix}{context_text}\n\n{full_system}"

        # 如果启用了 SelfRAG，追加自我评估指令
        if self.self_rag:
            full_system += self.self_rag.get_critique_instruction()

        messages = [{"role": "system", "content": full_system}]
        messages.extend(conversation)
        messages.append({"role": "user", "content": query})

        raw_reply = self.llm_client.call(messages, max_tokens=max_tokens + 200)

        if not raw_reply:
            return None, "部分支持", 3

        if self.self_rag:
            reply, is_sup, is_use = self.self_rag.critique_output(raw_reply)
            return reply, is_sup, is_use
        else:
            return raw_reply, "部分支持", 3

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.db_client.get_stats(collection_name=self.collection_name)
