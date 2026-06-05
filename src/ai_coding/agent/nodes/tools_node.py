"""Tools 节点工厂.

负责执行工具调用、处理 plan 阶段切换、追加探索软约束提示.
"""

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import MessagesState

from ai_coding.logger import get_logger

if TYPE_CHECKING:
    from ai_coding.agent.phase import PhaseController
    from ai_coding.tools.base import ToolRegistry

logger = get_logger(__name__)


def create_tools_node(tool_registry: "ToolRegistry", phase_ctrl: "PhaseController"):
    """创建 Tools 节点函数.

    Args:
        tool_registry: 工具注册表，用于查找和执行工具.
        phase_ctrl: 阶段控制器，用于跟踪探索/修改阶段.

    Returns:
        符合 LangGraph 节点签名的 callable.
    """

    def tools_node(state: MessagesState):
        """执行工具调用，处理 plan 状态，追加软约束提醒."""
        last_msg = state["messages"][-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return {"messages": []}

        tool_messages = []
        explore_steps = phase_ctrl.count_explore_steps(state["messages"])

        for tc in last_msg.tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {})
            tool_id = tc.get("id", "")

            # 确保 args 是 dict
            if hasattr(args, "dict"):
                args = args.dict()
            elif not isinstance(args, dict):
                args = dict(args)

            # 处理 plan 工具：标记进入修改阶段
            if name == "plan":
                phase_ctrl.on_plan_submitted(args.get("plan", ""))

            result = tool_registry.execute(name, args)

            # 软约束：探索工具的结果后追加提醒
            hint = phase_ctrl.build_post_explore_hint(name, explore_steps + 1)
            if hint:
                result += hint

            tool_messages.append(ToolMessage(content=result, tool_call_id=tool_id))

        return {"messages": tool_messages}

    return tools_node
