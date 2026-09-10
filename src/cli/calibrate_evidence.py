"""从标注样本中选择文本与图片证据阈值。"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.evaluation.evidence_calibrator import (
    EvidenceThresholdCalibrator,
    load_evidence_calibration_cases,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="校准RAG证据置信度阈值")
    parser.add_argument(
        "--dataset",
        default=os.getenv(
            "EVAL_EVIDENCE_DATASET", "./evaluation/evidence_calibration.jsonl"
        ),
    )
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    cases = load_evidence_calibration_cases(args.dataset)
    report = EvidenceThresholdCalibrator().calibrate(cases)

    output_path = args.output
    if not output_path:
        output_dir = Path(os.getenv("EVAL_OUTPUT_DIR", "./evaluation/results"))
        output_path = (
            output_dir / f"evidence_calibration_{datetime.now():%Y%m%d_%H%M%S}.json"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    text_result = report["modalities"]["text"]
    image_result = report["modalities"]["image"]
    summary = {
        "output": str(output_path.resolve()),
        "recommended_env": {
            "TUTOR_MIN_TEXT_EVIDENCE_SCORE": (
                text_result["recommended_threshold"] if text_result else None
            ),
            "TUTOR_MIN_IMAGE_EVIDENCE_SCORE": (
                image_result["recommended_threshold"] if image_result else None
            ),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
