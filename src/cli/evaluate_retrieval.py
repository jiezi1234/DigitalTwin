"""运行 baseline 与优化版人物对话检索对比评测。"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.evaluation.retrieval_evaluator import RetrievalEvaluator, load_evaluation_cases
from src.infrastructure.db_client import DBClient
from src.infrastructure.llm_client import LLMClient
from src.rag.bm25_retriever import BM25Retriever
from src.rag.llm_reranker import LLMReranker
from src.rag.metadata_filter import MetadataFilterBuilder
from src.rag.query_processor import QueryProcessor
from src.rag.rag_engine import RAGEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="比较基础相似度检索与查询改写 + 混合召回 + MMR"
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("EVAL_DATASET", "./evaluation/retrieval_queries.jsonl"),
    )
    parser.add_argument(
        "--collection", required=True, help="待评测的 Chroma collection"
    )
    parser.add_argument(
        "--persist-dir", default=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    )
    parser.add_argument("--k", type=int, default=int(os.getenv("EVAL_TOP_K", "5")))
    parser.add_argument(
        "--max-cases", type=int, default=None, help="仅评测前 N 条，用于小规模试跑"
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--no-query-rewriting",
        action="store_true",
        help="关闭优化组的查询改写，仅比较 MMR",
    )
    parser.add_argument(
        "--no-neighbor-expansion",
        action="store_true",
        help="关闭优化组的聊天时间邻域扩展",
    )
    parser.add_argument(
        "--no-hybrid-search",
        action="store_true",
        help="关闭优化组的 Dense/MMR + BM25 加权 RRF 混合召回",
    )
    parser.add_argument(
        "--no-reranking",
        action="store_true",
        help="关闭优化组的 LLM 候选重排",
    )
    parser.add_argument(
        "--no-metadata-filtering",
        action="store_true",
        help="关闭优化组的结构化时间范围过滤",
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    cases = load_evaluation_cases(args.dataset)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    db_client = DBClient(persist_dir=args.persist_dir)
    llm_client = LLMClient()
    rewriting_enabled = (
        os.getenv("EVAL_QUERY_REWRITING_ENABLED", "true").lower() == "true"
        and not args.no_query_rewriting
    )
    neighbor_expansion_enabled = (
        os.getenv("RAG_NEIGHBOR_EXPANSION_ENABLED", "true").lower() == "true"
        and not args.no_neighbor_expansion
    )
    hybrid_search_enabled = (
        os.getenv("RAG_HYBRID_SEARCH_ENABLED", "true").lower() == "true"
        and not args.no_hybrid_search
    )
    reranking_enabled = (
        os.getenv("RAG_RERANK_ENABLED", "true").lower() == "true"
        and not args.no_reranking
    )
    metadata_filtering_enabled = (
        os.getenv("RAG_METADATA_FILTERING_ENABLED", "true").lower() == "true"
        and not args.no_metadata_filtering
    )
    bm25_retriever = BM25Retriever(db_client) if hybrid_search_enabled else None
    reranker = (
        LLMReranker(
            llm_client=llm_client,
            model=os.getenv("RAG_RERANK_MODEL", "qwen-turbo"),
        )
        if reranking_enabled
        else None
    )
    engine = RAGEngine(
        db_client=db_client,
        lexical_retriever=bm25_retriever,
        reranker=reranker,
        metadata_filter_builder=(
            MetadataFilterBuilder(
                timezone_offset=os.getenv("RAG_TIMEZONE_OFFSET", "+08:00")
            )
            if metadata_filtering_enabled
            else None
        ),
    )
    processor = QueryProcessor(
        llm_client=llm_client,
        enable_coreference_resolution=True,
        enable_query_rewriting=rewriting_enabled,
        domain="persona",
    )

    def baseline_retriever(case, k):
        return db_client.search(
            query=case.query,
            collection_name=args.collection,
            k=k,
            use_mmr=False,
        )

    def optimized_retriever(case, k):
        semantic_results = engine.search(
            query=case.query,
            collection_name=args.collection,
            query_processor=processor,
            k=k,
            use_mmr=True,
            hybrid_search=hybrid_search_enabled,
            hybrid_candidates=int(os.getenv("RAG_HYBRID_CANDIDATES", "30")),
            rrf_k=int(os.getenv("RAG_HYBRID_RRF_K", "60")),
            dense_weight=float(os.getenv("RAG_DENSE_WEIGHT", "1.0")),
            bm25_weight=float(os.getenv("RAG_BM25_WEIGHT", "1.0")),
            rerank=reranking_enabled,
            rerank_candidates=int(os.getenv("RAG_RERANK_CANDIDATES", "20")),
            metadata_filtering=metadata_filtering_enabled,
            persona=case.persona,
        )
        if not neighbor_expansion_enabled:
            return semantic_results
        return engine.expand_chat_neighbors(
            semantic_results,
            collection_name=args.collection,
            window_size=int(os.getenv("RAG_NEIGHBOR_WINDOW", "2")),
            anchor_limit=int(os.getenv("RAG_NEIGHBOR_ANCHORS", "5")),
            max_results=int(os.getenv("RAG_NEIGHBOR_MAX_RESULTS", "30")),
        )

    optimized_features = ["mmr"]
    if rewriting_enabled:
        optimized_features.insert(0, "query_rewriting")
    if hybrid_search_enabled:
        optimized_features.append("hybrid_rrf")
    if reranking_enabled:
        optimized_features.append("llm_rerank")
    if metadata_filtering_enabled:
        optimized_features.append("metadata_filter")
    if neighbor_expansion_enabled:
        optimized_features.append("neighbors")

    report = RetrievalEvaluator(k=args.k).compare(
        cases,
        baseline_retriever=baseline_retriever,
        optimized_retriever=optimized_retriever,
        optimized_run_name="_".join(optimized_features),
    )
    output_path = args.output
    if not output_path:
        output_dir = Path(os.getenv("EVAL_OUTPUT_DIR", "./evaluation/results"))
        output_path = output_dir / f"retrieval_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "output": str(output_path.resolve()),
                "dataset_size": len(cases),
                "baseline": report["baseline"]["metrics"],
                "optimized": report["optimized"]["metrics"],
                "delta": report["delta"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
