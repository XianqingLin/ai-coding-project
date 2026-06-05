"""Agent 包.

包含 LangGraph ReAct Agent 核心、节点工厂、上下文管理、阶段控制与会话管理.
"""

from ai_coding.agent.context import ContextManager
from ai_coding.agent.core import LangGraphAgent
from ai_coding.agent.nodes import create_agent_node, create_should_continue, create_tools_node
from ai_coding.agent.phase import PhaseController
from ai_coding.agent.session import Session, SessionManager

__all__ = [
    "LangGraphAgent",
    "Session",
    "SessionManager",
    "ContextManager",
    "PhaseController",
    "create_agent_node",
    "create_tools_node",
    "create_should_continue",
]
