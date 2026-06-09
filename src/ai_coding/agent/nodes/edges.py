"""条件边工厂.

提供 LangGraph 的条件跳转逻辑.
"""

from langchain_core.messages import AIMessage
from langgraph.graph import END

from ai_coding.agent.state import AgentState


def create_should_continue():
    """创建条件判断函数：Agent 节点输出后是否需要调用工具.

    Returns:
        符合 LangGraph conditional_edges 签名的 callable.
    """

    def should_continue(state: AgentState):
        """判断是否需要继续调用工具."""
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "tools"
        return END

    return should_continue
