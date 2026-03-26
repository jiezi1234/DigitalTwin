"""
重建 PDF 多模态索引
使用方式:
python -m src.rebuild_multimodal_pdf_index "data/pdf/*.pdf"
"""

import argparse
import json

from src.api.config import Config
from src.infrastructure.db_client import DBClient
from src.infrastructure.multimodal_embedding_client import MultiModalEmbeddingClient
from src.services.multimodal_pdf_service import MultiModalPDFIndexService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="重建 PDF 多模态向量索引")
    parser.add_argument("pattern", help="PDF glob 模式，例如 data/pdf/*.pdf")
    parser.add_argument(
        "--persist-dir",
        default=Config.CHROMA_PERSIST_DIR,
        help="Chroma 持久化目录",
    )
    parser.add_argument(
        "--output-root",
        default=Config.PDF_EXPORT_ROOT,
        help="结构化导出根目录",
    )
    parser.add_argument(
        "--text-collection",
        default=Config.TUTOR_MM_TEXT_COLLECTION,
        help="文本块 collection 名称",
    )
    parser.add_argument(
        "--image-collection",
        default=Config.TUTOR_MM_IMAGE_COLLECTION,
        help="图片 collection 名称",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="导入前清空两个目标 collection",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    db_client = DBClient(
        persist_dir=args.persist_dir,
        embed_model=Config.EMBED_MODEL,
    )
    mm_client = MultiModalEmbeddingClient(model=Config.MM_EMBED_MODEL)
    service = MultiModalPDFIndexService(db_client=db_client, embedding_client=mm_client)

    if args.reset:
        db_client.delete_collection(args.text_collection)
        db_client.delete_collection(args.image_collection)

    results = service.index_pattern(
        pattern=args.pattern,
        output_root=args.output_root,
        text_collection=args.text_collection,
        image_collection=args.image_collection,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
