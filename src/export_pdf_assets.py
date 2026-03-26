"""
PDF 结构化导出脚本
使用方式:
python -m src.export_pdf_assets data/pdf/notes1_2022.pdf --output-dir output/pdf_export
"""

import argparse
import json
import os

from src.loaders.pdf_loader import PDFLoader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出 PDF 的文本块、图片和统一 JSON")
    parser.add_argument("pdf_path", help="PDF 文件路径")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="导出目录，默认使用 output/<pdf 文件名>",
    )
    parser.add_argument(
        "--json-filename",
        default="structured_content.json",
        help="统一 JSON 文件名",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
        output_dir = os.path.join("output", pdf_name)

    loader = PDFLoader(filepath=args.pdf_path, ocr_enabled=False)
    result = loader.export_structured(
        output_dir=output_dir,
        json_filename=args.json_filename,
    )

    print(json.dumps({
        "json_file": result["json_file"],
        "images_dir": result["images_dir"],
        "summary": result["summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
