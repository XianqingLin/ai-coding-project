"""Agent 对外暴露的标准事件类型.

所有前端（CLI、TUI、Web UI、测试脚本）都应通过这里定义的事件
与 AgentService 交互，而不是直接读取 LangGraphAgent 的内部状态.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Union


@dataclass(frozen=True)
class UserInputEvent:
    type: Literal["user_input"] = "user_input"
    text: str = ""


@dataclass(frozen=True)
class ThinkingStartEvent:
    type: Literal["thinking_start"] = "thinking_start"


@dataclass(frozen=True)
class ThinkingChunkEvent:
    type: Literal["thinking_chunk"] = "thinking_chunk"
    text: str = ""


@dataclass(frozen=True)
class ThinkingEndEvent:
    type: Literal["thinking_end"] = "thinking_end"


@dataclass(frozen=True)
class AssistantStartEvent:
    type: Literal["assistant_start"] = "assistant_start"


@dataclass(frozen=True)
class AssistantChunkEvent:
    type: Literal["assistant_chunk"] = "assistant_chunk"
    text: str = ""


@dataclass(frozen=True)
class AssistantEndEvent:
    type: Literal["assistant_end"] = "assistant_end"
    text: str = ""


@dataclass(frozen=True)
class ToolCallEvent:
    type: Literal["tool_call"] = "tool_call"
    name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservationEvent:
    type: Literal["observation"] = "observation"
    text: str = ""


@dataclass(frozen=True)
class ErrorEvent:
    type: Literal["error"] = "error"
    text: str = ""


AgentEvent = Union[
    UserInputEvent,
    ThinkingStartEvent,
    ThinkingChunkEvent,
    ThinkingEndEvent,
    AssistantStartEvent,
    AssistantChunkEvent,
    AssistantEndEvent,
    ToolCallEvent,
    ObservationEvent,
    ErrorEvent,
]
