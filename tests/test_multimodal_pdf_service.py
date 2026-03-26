import json
from pathlib import Path
from unittest.mock import MagicMock

from src.services.multimodal_pdf_service import MultiModalPDFIndexService
from src.infrastructure.document import Document


def test_build_text_and_image_documents(tmp_path):
    image_dir = tmp_path / "sample" / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "page_0001_img_001.png"
    image_path.write_bytes(b"fake-image")

    structured = {
        "source_file": "sample.pdf",
        "pages": [
            {
                "page": 1,
                "text_blocks": [
                    {
                        "block_index": 0,
                        "bbox": {"x0": 1, "y0": 2, "x1": 3, "y1": 4},
                        "content": "数据库系统基础",
                    }
                ],
                "images": [
                    {
                        "image_index": 0,
                        "bbox": {"x0": 10, "y0": 20, "x1": 30, "y1": 40},
                        "file": "images/page_0001_img_001.png",
                        "image_hash": "hash-1",
                        "extension": "png",
                        "width": 100,
                        "height": 80,
                    }
                ],
            }
        ],
    }

    service = MultiModalPDFIndexService(db_client=MagicMock(), embedding_client=MagicMock())
    text_docs = service.build_text_documents(structured)
    image_docs, image_paths = service.build_image_documents(structured, str(tmp_path / "sample"))

    assert len(text_docs) == 1
    assert text_docs[0].content == "数据库系统基础"
    assert json.loads(text_docs[0].metadata["bbox"])["x0"] == 1

    assert len(image_docs) == 1
    assert image_paths == [str(image_path)]
    assert image_docs[0].metadata["image_url"].startswith("/exports/")


def test_multimodal_tracking_skips_completed_batches(tmp_path):
    tracking_file = tmp_path / "tracking.json"

    db_client = MagicMock()
    db_client.bulk_import.return_value = 1

    embedding_client = MagicMock()
    embedding_client.embed_texts.return_value = [[0.1, 0.2]]

    service = MultiModalPDFIndexService(db_client=db_client, embedding_client=embedding_client)

    doc1 = Document(content="A", metadata={"page": 1}, doc_id="doc-1")
    doc2 = Document(content="B", metadata={"page": 2}, doc_id="doc-2")

    tracking_file.write_text(
        json.dumps({"completed_text_ids": ["doc-1"], "completed_image_ids": []}),
        encoding="utf-8",
    )

    imported_count, skipped_count = service._import_text_batches(
        collection_name="test_collection",
        documents=[doc1, doc2],
        batch_size=1,
        max_workers=1,
        tracking_file=str(tracking_file),
    )

    assert imported_count == 1
    assert skipped_count == 1
    embedding_client.embed_texts.assert_called_once_with(["B"])
