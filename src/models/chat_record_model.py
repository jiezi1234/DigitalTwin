"""
标准化聊天记录数据模型

支持多种数据源格式，将其统一映射到标准字段
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ChatRecord:
    """标准化聊天记录数据结构"""

    # ──── 必须字段（所有数据源都必须有）────────────
    talker: str              # 发送者名称
    msg: str                 # 消息内容
    is_sender: int           # 是否是 self 发送（0=他人, 1=自己）

    # ──── 可选字段（某些数据源可能没有）─────────────
    CreateTime: Optional[str] = None      # 创建时间（ISO8601 或其他格式）
    MsgSvrID: Optional[str] = None        # 消息服务器 ID
    type_name: Optional[str] = None       # 消息类型（text, image, voice, link等）
    src: Optional[str] = None             # 消息来源
    room_name: Optional[str] = None       # 所属群组名称
    is_forward: Optional[int] = None      # 是否转发（0=否, 1=是）
    id: Optional[int] = None              # 行号或消息 ID

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatRecord":
        """从字典创建 ChatRecord，自动处理字段映射"""

        # 验证必须字段
        talker = str(data.get("talker", "")).strip()
        msg = str(data.get("msg", "")).strip()
        is_sender_raw = data.get("is_sender", 0)

        if not talker:
            raise ValueError("Missing required field: talker")
        if not msg:
            raise ValueError("Missing required field: msg")

        # 解析 is_sender（可能是字符串"0"/"1"或整数）
        try:
            is_sender = int(is_sender_raw) if is_sender_raw is not None else 0
        except (ValueError, TypeError):
            is_sender = 0

        # 提取可选字段，允许值为 None
        CreateTime = data.get("CreateTime") or None
        MsgSvrID = data.get("MsgSvrID") or None
        type_name = data.get("type_name") or None
        src = data.get("src") or None
        room_name = data.get("room_name") or None
        id_val = data.get("id")

        # 解析 is_forward
        is_forward = None
        if "is_forward" in data and data["is_forward"] is not None:
            try:
                is_forward = int(data.get("is_forward", 0))
            except (ValueError, TypeError):
                is_forward = None

        return cls(
            talker=talker,
            msg=msg,
            is_sender=is_sender,
            CreateTime=CreateTime,
            MsgSvrID=MsgSvrID,
            type_name=type_name,
            src=src,
            room_name=room_name,
            is_forward=is_forward,
            id=id_val,
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，仅包含非 None 的字段"""
        result = {
            "talker": self.talker,
            "msg": self.msg,
            "is_sender": self.is_sender,
        }

        # 添加可选字段（仅当值不为 None 时）
        if self.CreateTime is not None:
            result["CreateTime"] = self.CreateTime
        if self.MsgSvrID is not None:
            result["MsgSvrID"] = self.MsgSvrID
        if self.type_name is not None:
            result["type_name"] = self.type_name
        if self.src is not None:
            result["src"] = self.src
        if self.room_name is not None:
            result["room_name"] = self.room_name
        if self.is_forward is not None:
            result["is_forward"] = self.is_forward
        if self.id is not None:
            result["id"] = self.id

        return result

    def has_time(self) -> bool:
        """检查是否有时间信息"""
        return self.CreateTime is not None

    def has_room(self) -> bool:
        """检查是否有群组信息"""
        return self.room_name is not None and self.room_name != ""

    def msg_type(self) -> str:
        """获取消息类型，默认为 'text'"""
        return self.type_name or "text"


class ChatRecordSchema:
    """数据格式检测和验证工具"""

    # 已知的数据源格式
    FORMAT_WECHAT_STANDARD = "wechat_standard"  # 完整格式：lccc_chat*, zhenhuan_chat
    FORMAT_SIMPLE = "simple"                    # 简洁格式：daiyu_full_sessions
    FORMAT_UNKNOWN = "unknown"

    @staticmethod
    def detect_format(headers: list) -> str:
        """根据 CSV 列名检测数据格式"""
        headers_set = set(h.strip() for h in headers if h)

        # 检查是否包含完整的 WeChat 标准格式字段
        standard_required = {"id", "MsgSvrID", "type_name", "is_sender", "talker", "msg", "CreateTime"}
        if standard_required.issubset(headers_set):
            return ChatRecordSchema.FORMAT_WECHAT_STANDARD

        # 检查是否是简洁格式
        simple_required = {"is_sender", "talker", "msg"}
        if simple_required.issubset(headers_set):
            return ChatRecordSchema.FORMAT_SIMPLE

        return ChatRecordSchema.FORMAT_UNKNOWN

    @staticmethod
    def get_field_info() -> Dict[str, Dict[str, Any]]:
        """获取所有字段的元数据"""
        return {
            # 必须字段
            "talker": {
                "type": "string",
                "required": True,
                "description": "消息发送者的名称",
                "example": "林黛玉 / self / 丫头们",
            },
            "msg": {
                "type": "string",
                "required": True,
                "description": "消息内容（不能为空或仅是emoji）",
                "example": "你好吗？",
            },
            "is_sender": {
                "type": "int (0/1)",
                "required": True,
                "description": "是否是 self 发送：0=他人, 1=自己",
                "example": "0 / 1",
            },

            # 可选字段
            "CreateTime": {
                "type": "string (ISO8601 or datetime)",
                "required": False,
                "description": "消息创建时间",
                "example": "2024-01-15 10:30:45",
            },
            "MsgSvrID": {
                "type": "string",
                "required": False,
                "description": "微信消息服务器 ID（用于去重和追踪）",
                "example": "1234567890",
            },
            "type_name": {
                "type": "string",
                "required": False,
                "description": "消息类型（text, image, voice, video, link等）",
                "example": "text / image / voice",
            },
            "src": {
                "type": "string",
                "required": False,
                "description": "消息来源（如导出工具名）",
                "example": "WeChat",
            },
            "room_name": {
                "type": "string",
                "required": False,
                "description": "所属群组名称（空或空字符串表示私聊）",
                "example": "红楼梦讨论组 / null",
            },
            "is_forward": {
                "type": "int (0/1)",
                "required": False,
                "description": "是否转发的消息：0=否, 1=是",
                "example": "0 / 1",
            },
            "id": {
                "type": "int or string",
                "required": False,
                "description": "行号或消息唯一标识符",
                "example": "1 / 12345",
            },
        }
