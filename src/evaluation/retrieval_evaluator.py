"""检索评测指标与 JSONL 数据集加载。"""

import json
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

SearchResult = Tuple[str, Dict[str, Any], float]
Retriever = Callable[["EvaluationCase", int], List[SearchResult]]


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    query: str
    relevant: List[Dict[str, Any]]
    persona: Dict[str, Any] = field(default_factory=dict)


def load_evaluation_cases(path: str) -> List[EvaluationCase]:
    """加载并校验一行一个样本的 JSONL 标注集。"""
    cases: List[EvaluationCase] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"评测集第 {line_number} 行不是合法 JSON: {exc}"
                ) from exc

            query = str(payload.get("query", "")).strip()
            relevant = payload.get("relevant")
            if not query:
                raise ValueError(f"评测集第 {line_number} 行缺少 query")
            if not isinstance(relevant, list) or not relevant:
                raise ValueError(f"评测集第 {line_number} 行 relevant 必须是非空数组")
            if not all(isinstance(label, dict) and label for label in relevant):
                raise ValueError(f"评测集第 {line_number} 行包含无效 relevant 标签")

            cases.append(
                EvaluationCase(
                    case_id=str(payload.get("id") or f"case-{line_number}"),
                    query=query,
                    relevant=relevant,
                    persona=payload.get("persona") or {},
                )
            )

    if not cases:
        raise ValueError("评测集为空")
    return cases


class RetrievalEvaluator:
    """计算 Hit Rate、MRR、Recall 和查询延迟。"""

    def __init__(self, k: int = 5):
        if k < 1:
            raise ValueError("k 必须大于 0")
        self.k = k

    @staticmethod
    def _matches(content: str, metadata: Dict[str, Any], label: Dict[str, Any]) -> bool:
        for key, expected in label.items():
            if key == "content_contains":
                if str(expected) not in content:
                    return False
                continue
            actual = metadata.get(key)
            if str(actual) != str(expected):
                return False
        return True

    def evaluate(
        self,
        cases: Iterable[EvaluationCase],
        retriever: Retriever,
        run_name: str,
    ) -> Dict[str, Any]:
        case_list = list(cases)
        if not case_list:
            raise ValueError("没有可评测样本")

        hit_count = 0
        reciprocal_ranks: List[float] = []
        recalls: List[float] = []
        context_precisions: List[float] = []
        latencies_ms: List[float] = []
        details: List[Dict[str, Any]] = []

        for case in case_list:
            started = time.perf_counter()
            results = retriever(case, self.k)[: self.k]
            latency_ms = (time.perf_counter() - started) * 1000
            latencies_ms.append(latency_ms)

            matched_label_indices = set()
            first_relevant_rank: Optional[int] = None
            relevant_result_count = 0
            for rank, (content, metadata, _) in enumerate(results, 1):
                result_is_relevant = False
                for label_index, label in enumerate(case.relevant):
                    if self._matches(content, metadata or {}, label):
                        matched_label_indices.add(label_index)
                        result_is_relevant = True
                        if first_relevant_rank is None:
                            first_relevant_rank = rank
                relevant_result_count += int(result_is_relevant)

            hit = first_relevant_rank is not None
            hit_count += int(hit)
            reciprocal_ranks.append(
                1.0 / first_relevant_rank if first_relevant_rank else 0.0
            )
            recalls.append(len(matched_label_indices) / len(case.relevant))
            context_precision = relevant_result_count / len(results) if results else 0.0
            context_precisions.append(context_precision)
            details.append(
                {
                    "id": case.case_id,
                    "hit": hit,
                    "first_relevant_rank": first_relevant_rank,
                    "matched_labels": len(matched_label_indices),
                    "relevant_labels": len(case.relevant),
                    "returned": len(results),
                    "relevant_results": relevant_result_count,
                    "context_precision": context_precision,
                    "latency_ms": round(latency_ms, 3),
                }
            )

        return {
            "run_name": run_name,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_size": len(case_list),
            "k": self.k,
            "metrics": {
                f"hit_rate@{self.k}": hit_count / len(case_list),
                f"mrr@{self.k}": statistics.fmean(reciprocal_ranks),
                f"recall@{self.k}": statistics.fmean(recalls),
                f"context_precision@{self.k}": statistics.fmean(context_precisions),
                "latency_ms_mean": statistics.fmean(latencies_ms),
                "latency_ms_p50": self._percentile(latencies_ms, 0.50),
                "latency_ms_p95": self._percentile(latencies_ms, 0.95),
            },
            "cases": details,
        }

    @staticmethod
    def _percentile(values: List[float], quantile: float) -> float:
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        index = (len(ordered) - 1) * quantile
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        weight = index - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    def compare(
        self,
        cases: Iterable[EvaluationCase],
        baseline_retriever: Retriever,
        optimized_retriever: Retriever,
        optimized_run_name: str = "query_rewriting_mmr",
    ) -> Dict[str, Any]:
        case_list = list(cases)
        baseline = self.evaluate(case_list, baseline_retriever, "baseline_similarity")
        optimized = self.evaluate(case_list, optimized_retriever, optimized_run_name)
        metric_keys = (
            f"hit_rate@{self.k}",
            f"mrr@{self.k}",
            f"recall@{self.k}",
            f"context_precision@{self.k}",
        )
        return {
            "baseline": baseline,
            "optimized": optimized,
            "delta": {
                **{
                    key: optimized["metrics"][key] - baseline["metrics"][key]
                    for key in metric_keys
                },
                "improved_cases": [
                    optimized_case["id"]
                    for baseline_case, optimized_case in zip(
                        baseline["cases"], optimized["cases"]
                    )
                    if not baseline_case["hit"] and optimized_case["hit"]
                ],
                "regressed_cases": [
                    optimized_case["id"]
                    for baseline_case, optimized_case in zip(
                        baseline["cases"], optimized["cases"]
                    )
                    if baseline_case["hit"] and not optimized_case["hit"]
                ],
            },
        }
