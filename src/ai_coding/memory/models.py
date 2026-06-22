"""记忆数据模型."""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryType(str, Enum):
    """记忆类型."""

    PREFERENCE = "preference"  # 用户偏好，例如语言、风格
    RULE = "rule"  # 项目规则，例如代码规范
    FACT = "fact"  # 项目事实，例如文件位置
    ARCHITECTURE = "architecture"  # 架构决策


class MemoryScope(str, Enum):
    """记忆作用域."""

    USER = "user"  # 跨项目生效
    PROJECT = "project"  # 仅在当前项目生效


def _parse_scope(value: Any) -> MemoryScope:
    """解析作用域，未知值回退为 PROJECT."""
    try:
        return MemoryScope(value)
    except ValueError:
        return MemoryScope.PROJECT


@dataclass
class MemoryEntry:
    """结构化记忆条目.

    Attributes:
        content: 记忆文本内容.
        type: 记忆类型.
        scope: 记忆作用域，第一版默认 project.
        confidence: 置信度 0.0~1.0，越高越稳定.
        observation_count: 被观察/确认的次数.
        explicit: 是否来自用户显式声明.
        source_sessions: 来源会话 ID 列表.
        id: 唯一标识.
        created_at: 创建时间戳.
        updated_at: 最后更新时间戳.
        expires_at: 过期时间戳，None 表示不过期.
        invalidated: 是否被用户显式否定或淘汰.
    """

    content: str
    type: MemoryType = MemoryType.FACT
    scope: MemoryScope = MemoryScope.PROJECT
    confidence: float = 0.5
    observation_count: int = 1
    explicit: bool = False
    source_sessions: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    invalidated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典."""
        return {
            "id": self.id,
            "content": self.content,
            "type": self.type.value,
            "scope": self.scope.value,
            "confidence": self.confidence,
            "observation_count": self.observation_count,
            "explicit": self.explicit,
            "source_sessions": list(self.source_sessions),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "invalidated": self.invalidated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """从字典反序列化."""
        return cls(
            id=data.get("id") or str(uuid.uuid4()),
            content=data["content"],
            type=MemoryType(data.get("type", MemoryType.FACT.value)),
            scope=_parse_scope(data.get("scope", MemoryScope.PROJECT.value)),
            confidence=float(data.get("confidence", 0.5)),
            observation_count=int(data.get("observation_count", 1)),
            explicit=bool(data.get("explicit", False)),
            source_sessions=list(data.get("source_sessions", [])),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            expires_at=data.get("expires_at"),
            invalidated=bool(data.get("invalidated", False)),
        )
