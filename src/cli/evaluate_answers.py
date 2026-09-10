"""运行回答级引用、拒答与忠实度评测。"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.evaluation.answer_evaluator import (
    AnswerEvaluator,
    LLMGroundednessJudge,
    load_answer_cases,
)
from src.infrastructure.llm_client import LLMClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评测RAG回答质量")
    parser.add_argument(
        "--dataset",
        default=os.getenv("EVAL_ANSWER_DATASET", "./evaluation/answer_cases.jsonl"),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="额外调用模型评估回答相对上下文的忠实度",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("EVAL_JUDGE_MODEL", "qwen-turbo"),
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    cases = load_answer_cases(args.dataset)
    judge = None
    if args.llm_judge:
        judge = LLMGroundednessJudge(LLMClient(), model=args.judge_model)

    report = AnswerEvaluator(groundedness_judge=judge).evaluate(cases)
    output_path = args.output
    if not output_path:
        output_dir = Path(os.getenv("EVAL_OUTPUT_DIR", "./evaluation/results"))
        output_path = output_dir / f"answer_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(output_path.resolve()), **report["metrics"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
