from src.rag.context_builder import ContextBuilder


def test_context_builder_deduplicates_and_renumbers_textbook_results():
    builder = ContextBuilder(dedup_threshold=0.9)
    results = [
        ("事务具有原子性", {"source_file": "book.pdf", "page": 1}, 0.9),
        ("事务具有原子性。", {"source_file": "scan.pdf", "page": 2}, 0.8),
        ("隔离性用于控制并发", {"source_file": "book.pdf", "page": 3}, 0.7),
    ]

    built = builder.build(results, format_type="textbook")

    assert built.selected_count == 2
    assert built.duplicate_count == 1
    assert [item[0] for item in built.selected_results] == [
        "事务具有原子性",
        "隔离性用于控制并发",
    ]
    assert "[1]【book.pdf > 第1页】" in built.text
    assert "[2]【book.pdf > 第3页】" in built.text
    assert "scan.pdf" not in built.text


def test_context_builder_respects_total_and_per_record_budget():
    builder = ContextBuilder(max_record_chars=60, min_remaining_chars=20)
    results = [
        ("第一段。" + "甲" * 150, {"talker": "张三"}, 0.9),
        ("第二段。" + "乙" * 150, {"talker": "李四"}, 0.8),
    ]

    built = builder.build(results, max_context_length=130, format_type="chat")

    assert built.selected_count == 2
    assert built.truncated_count == 2
    assert built.used_chars <= 130
    assert built.text.count("…") == 2
    assert all(len(item[0]) <= 60 for item in built.selected_results)


def test_context_builder_does_not_repeat_existing_chat_speaker_prefix():
    builder = ContextBuilder()

    built = builder.build(
        [("张三: 已经带前缀", {"talker": "张三", "chat_time": 100}, 0.9)],
        format_type="chat",
    )

    assert built.text == "[100] 张三: 已经带前缀"
    assert "张三: 张三:" not in built.text


def test_context_builder_returns_empty_result_for_zero_budget():
    built = ContextBuilder().build([("内容", {}, 0.9)], max_context_length=0)

    assert built.text == ""
    assert built.selected_count == 0
    assert built.used_chars == 0


def test_context_builder_keeps_short_record_in_small_budget():
    built = ContextBuilder(min_remaining_chars=80).build(
        [("短内容", {}, 0.9)],
        max_context_length=20,
        include_metadata=False,
    )

    assert built.text == "短内容"
    assert built.selected_count == 1
