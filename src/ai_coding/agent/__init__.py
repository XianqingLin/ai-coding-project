"""Agent 包.

包含 LangGraph ReAct Agent 核心、节点工厂、上下文管理、阶段控制、
会话管理以及对外统一的 AgentService 接口.
"""

from ai_coding.agent.core import LangGraphAgent
from ai_coding.agent.events import (
    AgentEvent,
    AssistantChunkEvent,
    AssistantEndEvent,
    AssistantStartEvent,
    ErrorEvent,
    ObservationEvent,
    ThinkingChunkEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallEvent,
    UserInputEvent,
)
from ai_coding.agent.nodes import (
    create_llm_node,
    create_should_continue,
    create_tools_node,
)
from ai_coding.agent.service import AgentService
from ai_coding.agent.session import Session, SessionManager

__all__ = [
    "AgentService",
    "AgentEvent",
    "AssistantChunkEvent",
    "AssistantEndEvent",
    "AssistantStartEvent",
    "ErrorEvent",
    "LangGraphAgent",
    "ObservationEvent",
    "Session",
    "SessionManager",
    "ThinkingChunkEvent",
    "ThinkingEndEvent",
    "ThinkingStartEvent",
    "ToolCallEvent",
    "UserInputEvent",
    "create_llm_node",
    "create_tools_node",
    "create_should_continue",
]
