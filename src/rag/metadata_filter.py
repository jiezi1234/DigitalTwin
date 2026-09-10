"""将结构化查询约束转换为 Chroma metadata filter。"""

import logging
import re
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, Optional

from src.rag.query_processor import QueryUnderstanding

logger = logging.getLogger(__name__)
_OFFSET_PATTERN = re.compile(r"^([+-])(\d{2}):?(\d{2})$")


class MetadataFilterBuilder:
    """为人物聊天索引构建可验证的时间范围过滤条件。"""

    def __init__(self, timezone_offset: str = "+08:00"):
        self.timezone = self._parse_timezone(timezone_offset)

    @staticmethod
    def _parse_timezone(value: str) -> timezone:
        match = _OFFSET_PATTERN.match(str(value).strip())
        if not match:
            logger.warning("无效时区偏移 %s，回退到 +08:00", value)
            return timezone(timedelta(hours=8))
        sign = 1 if match.group(1) == "+" else -1
        hours = int(match.group(2))
        minutes = int(match.group(3))
        if hours > 23 or minutes > 59:
            logger.warning("无效时区偏移 %s，回退到 +08:00", value)
            return timezone(timedelta(hours=8))
        return timezone(sign * timedelta(hours=hours, minutes=minutes))

    def build(
        self, understanding: Optional[QueryUnderstanding]
    ) -> Optional[Dict[str, Any]]:
        """把 ISO 时间范围转换为基于秒级 ``chat_time`` 的 Chroma 条件。"""
        if understanding is None or not understanding.time_range:
            return None

        start = self._parse_boundary(understanding.time_range.get("start"), False)
        end = self._parse_boundary(understanding.time_range.get("end"), True)
        if start is None and end is None:
            return None
        if start is not None and end is not None and start > end:
            logger.warning("查询时间范围起点晚于终点，忽略元数据过滤")
            return None

        conditions = []
        if start is not None:
            conditions.append({"chat_time": {"$gte": int(start.timestamp())}})
        if end is not None:
            conditions.append({"chat_time": {"$lte": int(end.timestamp())}})
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _parse_boundary(
        self, raw: Optional[str], end_of_day: bool
    ) -> Optional[datetime]:
        value = str(raw or "").strip()
        if not value:
            return None
        try:
            normalized = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
            if "T" not in normalized and " " not in normalized:
                boundary_time = time.max if end_of_day else time.min
                parsed = datetime.combine(parsed.date(), boundary_time)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=self.timezone)
            return parsed
        except ValueError:
            logger.warning("无法解析查询时间边界 %s，忽略该边界", value)
            return None
