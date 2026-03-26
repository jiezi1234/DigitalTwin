import json
import os

import pytest

from src.loaders.pdf_loader import PDFLoader

TEST_PDF = "data/pdf/test.pdf"


def test_pdf_loader_export_structured(tmp_path):
    """结构化导出应生成 JSON 清单和图片文件"""
    if not os.path.exists(TEST_PDF):
        pytest.skip(f"测试文件不存在: {TEST_PDF}")

    output_dir = tmp_path / "pdf_export"
    loader = PDFLoader(filepath=TEST_PDF, ocr_enabled=False, max_workers=1)

    result = loader.export_structured(str(output_dir))

    json_path = output_dir / "structured_content.json"
    images_dir = output_dir / "images"

    assert json_path.exists()
    assert images_dir.exists()
    assert result["summary"]["page_count"] > 0
    assert result["summary"]["text_block_count"] > 0
    assert result["summary"]["image_count"] > 0

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["source"] == "pdf"
    assert data["source_file"] == os.path.basename(TEST_PDF)
    assert data["summary"]["page_count"] == len(data["pages"])

    first_page = data["pages"][0]
    assert "text_blocks" in first_page
    assert "images" in first_page

    first_text_page = next(page for page in data["pages"] if page["text_blocks"])
    first_image_page = next(page for page in data["pages"] if page["images"])

    first_text_block = first_text_page["text_blocks"][0]
    assert {"block_index", "bbox", "content"} <= set(first_text_block.keys())
    assert {"x0", "y0", "x1", "y1"} <= set(first_text_block["bbox"].keys())
    assert first_text_block["content"]

    first_image = first_image_page["images"][0]
    assert {"image_index", "bbox", "file", "extension", "image_hash"} <= set(first_image.keys())
    assert (output_dir / first_image["file"]).exists()
