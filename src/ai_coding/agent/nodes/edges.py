"""条件边工厂.

提供 LangGraph 的条件跳转逻辑.
"""

from langchain_core.messages import AIMessage
from langgraph.graph import END

from ai_coding.agent.state import AgentState
from ai_coding.tools.base import ToolRegistry


def create_should_continue(tool_registry: ToolRegistry, auto_approve: bool = False):
    """创建条件判断函数：Agent 节点输出后是否需要调用工具，以及是否需要授权.

    Args:
        tool_registry: 工具注册表，用于查询工具是否需要授权.
        auto_approve: 是否自动批准所有工具调用（非交互模式）.

    Returns:
        符合 LangGraph conditional_edges 签名的 callable.
    """

    def should_continue(state: AgentState):
        """判断是否需要继续调用工具，以及是否需要先经过授权."""
        last_msg = state["messages"][-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return END

        if auto_approve:
            return "tools"

        globally_approved = set(state.get("globally_approved_tools", []))

        for tc in last_msg.tool_calls:
            name = tc.get("name", "")
            try:
                tool = tool_registry.get(name)
            except KeyError:
                continue

            if getattr(tool, "requires_approval", False) and name not in globally_approved:
                return "approval"

        return "tools"

    return should_continue
