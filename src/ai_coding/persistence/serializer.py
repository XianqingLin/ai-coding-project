"""状态序列化与反序列化模块.

负责 LangChain 消息和 AgentState 的 JSON 序列化/反序列化.
"""

import json
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# 消息类型到类的映射
_MSG_TYPE_MAP = {
    "human": HumanMessage,
    "ai": AIMessage,
    "tool": ToolMessage,
    "system": SystemMessage,
}


def _serialize_message(msg: BaseMessage) -> dict:
    """将单条 LangChain 消息序列化为字典."""
    data = msg.model_dump()
    # 确保 type 字段存在
    if "type" not in data:
        if isinstance(msg, HumanMessage):
            data["type"] = "human"
        elif isinstance(msg, AIMessage):
            data["type"] = "ai"
        elif isinstance(msg, ToolMessage):
            data["type"] = "tool"
        elif isinstance(msg, SystemMessage):
            data["type"] = "system"
    return data


def _deserialize_message(data: dict) -> BaseMessage:
    """将字典反序列化为 LangChain 消息."""
    msg_type = data.get("type", "")
    cls = _MSG_TYPE_MAP.get(msg_type)
    if cls is None:
        # 兜底：尝试根据字段推断
        if "tool_call_id" in data:
            cls = ToolMessage
        elif "tool_calls" in data:
            cls = AIMessage
        elif data.get("role") == "system":
            cls = SystemMessage
        else:
            cls = HumanMessage

    # 过滤掉 LangChain 内部字段，保留核心字段
    core_fields = {
        "content",
        "type",
        "name",
        "tool_call_id",
        "tool_calls",
        "id",
        "additional_kwargs",
    }
    kwargs = {
        k: v for k, v in data.items() if k in core_fields or not k.startswith("_")
    }

    return cls(**kwargs)


def serialize_messages(messages: Sequence[BaseMessage]) -> List[dict]:
    """序列化消息列表."""
    return [_serialize_message(m) for m in messages]


def deserialize_messages(data: List[dict]) -> List[BaseMessage]:
    """反序列化消息列表."""
    return [_deserialize_message(d) for d in data]


def serialize_state(state: Optional[Dict[str, Any]]) -> dict:
    """序列化 AgentState.

    Args:
        state: AgentState 字典.

    Returns:
        可 JSON 序列化的字典.

    """
    if state is None:
        return {"__version__": 1, "messages": []}

    result: Dict[str, Any] = {"__version__": 1}

    # messages
    msgs = state.get("messages", [])
    result["messages"] = serialize_messages(msgs)

    # file_snapshots
    result["file_snapshots"] = dict(state.get("file_snapshots", {}))

    # todos
    result["todos"] = [dict(t) for t in state.get("todos", [])]

    # globally_approved_tools
    result["globally_approved_tools"] = list(state.get("globally_approved_tools", []))

    # background_tasks
    result["background_tasks"] = [dict(t) for t in state.get("background_tasks", [])]

    # plan_mode
    result["plan_mode"] = bool(state.get("plan_mode", False))

    # plan_file_path
    result["plan_file_path"] = str(state.get("plan_file_path", ""))

    # sub_agents
    result["sub_agents"] = [dict(s) for s in state.get("sub_agents", [])]

    return result


def deserialize_state(data: dict) -> Dict[str, Any]:
    """反序列化 AgentState.

    Args:
        data: 序列化后的字典.

    Returns:
        AgentState 字典.

    """
    result: Dict[str, Any] = {}

    # messages
    result["messages"] = deserialize_messages(data.get("messages", []))

    # file_snapshots
    result["file_snapshots"] = dict(data.get("file_snapshots", {}))

    # todos
    result["todos"] = [dict(t) for t in data.get("todos", [])]

    # globally_approved_tools
    result["globally_approved_tools"] = list(data.get("globally_approved_tools", []))

    # background_tasks
    result["background_tasks"] = [dict(t) for t in data.get("background_tasks", [])]

    # plan_mode
    result["plan_mode"] = bool(data.get("plan_mode", False))

    # plan_file_path
    result["plan_file_path"] = str(data.get("plan_file_path", ""))

    # sub_agents
    result["sub_agents"] = [dict(s) for s in data.get("sub_agents", [])]

    return result


def state_to_json(state: Optional[Dict[str, Any]]) -> str:
    """将 AgentState 序列化为 JSON 字符串."""
    return json.dumps(serialize_state(state), ensure_ascii=False, indent=2)


def state_from_json(text: str) -> Dict[str, Any]:
    """从 JSON 字符串反序列化 AgentState."""
    return deserialize_state(json.loads(text))
