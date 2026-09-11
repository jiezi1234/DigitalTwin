from src.loaders.pdf_chunker import StructureAwarePDFChunker


def test_structure_chunker_preserves_heading_and_isolates_table():
    chunker = StructureAwarePDFChunker(target_chars=40, max_chars=80, overlap_blocks=0)
    blocks = [
        {
            "block_index": 0,
            "content": "第1章 数据库基础",
            "content_type": "heading",
            "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 20},
        },
        {
            "block_index": 1,
            "content": "数据库系统由数据库、数据库管理系统等部分组成。",
            "content_type": "paragraph",
            "bbox": {"x0": 0, "y0": 30, "x1": 100, "y1": 50},
        },
        {
            "block_index": 2,
            "content": "模型 | 特点\n关系模型 | 二维表",
            "content_type": "table",
            "bbox": {"x0": 0, "y0": 60, "x1": 100, "y1": 100},
        },
    ]

    chunks = chunker.chunk_blocks(blocks)

    assert chunks[0].content.startswith("第1章 数据库基础\n")
    assert chunks[0].block_indices == [1]
    assert chunks[1].content.endswith("关系模型 | 二维表")
    assert chunks[1].content_types == ["table"]
    assert chunks[1].bbox["y1"] == 100.0


def test_structure_chunker_splits_long_paragraph_on_sentence_boundaries():
    chunker = StructureAwarePDFChunker(target_chars=20, max_chars=30, overlap_blocks=0)
    text = "第一句话用于说明数据库。第二句话继续解释事务。第三句话描述索引。"

    chunks = chunker.split_text(text)

    assert len(chunks) >= 2
    assert all(len(chunk) <= 30 for chunk in chunks)
    assert "".join(chunks) == text


def test_structure_chunker_detects_caption_list_and_heading():
    assert StructureAwarePDFChunker.classify_block("图2-1 系统架构") == "caption"
    assert StructureAwarePDFChunker.classify_block("1.2 关系模型") == "heading"
    assert StructureAwarePDFChunker.classify_block("- 原子性") == "list"
    assert (
        StructureAwarePDFChunker.classify_block(
            "Database Systems", font_size=28, reference_font_size=16
        )
        == "heading"
    )


def test_structure_chunker_combines_consecutive_visual_heading_blocks():
    chunker = StructureAwarePDFChunker(overlap_blocks=0)

    chunks = chunker.chunk_blocks(
        [
            {"block_index": 0, "content": "Database", "content_type": "heading"},
            {"block_index": 1, "content": "Systems", "content_type": "heading"},
            {"block_index": 2, "content": "课程内容", "content_type": "paragraph"},
        ]
    )

    assert chunks[0].heading == "Database Systems"
    assert chunks[0].content == "Database Systems\n课程内容"
