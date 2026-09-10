from datetime import datetime, timedelta, timezone

from src.rag.metadata_filter import MetadataFilterBuilder
from src.rag.query_processor import QueryUnderstanding


def test_metadata_filter_builds_inclusive_date_range():
    builder = MetadataFilterBuilder("+08:00")
    understanding = QueryUnderstanding(
        original_query="去年说过什么",
        standalone_query="2025年聊天",
        time_range={"start": "2025-01-01", "end": "2025-12-31"},
    )

    where = builder.build(understanding)

    china_timezone = timezone(timedelta(hours=8))
    expected_start = int(datetime(2025, 1, 1, tzinfo=china_timezone).timestamp())
    expected_end = int(
        datetime(2025, 12, 31, 23, 59, 59, 999999, tzinfo=china_timezone).timestamp()
    )
    assert where == {
        "$and": [
            {"chat_time": {"$gte": expected_start}},
            {"chat_time": {"$lte": expected_end}},
        ]
    }


def test_metadata_filter_supports_one_sided_range():
    understanding = QueryUnderstanding(
        original_query="九月之后",
        standalone_query="九月之后",
        time_range={"start": "2026-09-01"},
    )

    where = MetadataFilterBuilder("+00:00").build(understanding)

    assert where == {
        "chat_time": {
            "$gte": int(datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp())
        }
    }


def test_metadata_filter_ignores_invalid_or_reversed_range():
    builder = MetadataFilterBuilder("+08:00")
    invalid = QueryUnderstanding("q", "q", time_range={"start": "去年"})
    reversed_range = QueryUnderstanding(
        "q",
        "q",
        time_range={"start": "2026-12-01", "end": "2026-01-01"},
    )

    assert builder.build(invalid) is None
    assert builder.build(reversed_range) is None
