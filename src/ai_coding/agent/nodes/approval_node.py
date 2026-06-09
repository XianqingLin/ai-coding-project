"""Approval Gate 节点工厂.

在 LLM 输出 tool_calls 之后、tools 节点执行之前插入，
负责拦截需要用户授权的写操作工具，提供交互式确认。
"""

import sys
from typing import Dict, List, Set

from langchain_core.messages import AIMessage, ToolMessage

from ai_coding.agent.state import AgentState
from ai_coding.logger import get_logger
from ai_coding.tools.base import ToolRegistry

logger = get_logger(__name__)


def create_approval_gate(tool_registry: ToolRegistry, interactive: bool = True):
    """创建 Approval Gate 节点函数.

    检查 LLM 输出的 tool_calls 中是否有需要用户授权的调用。
    - 交互模式下逐个询问：y(单次) / n(拒绝) / a(全局授权) / q(退出)
    - 非交互模式下直接拒绝所有未授权调用

    被拒绝的调用会生成 ToolMessage 返回给 LLM，让 Agent 重新规划。
    被批准的调用不做任何处理，由下游 tools 节点正常执行。

    Args:
        tool_registry: 工具注册表，用于查询工具是否需要授权.
        interactive: 是否为交互模式（False 时直接拒绝未授权调用）.

    Returns:
        符合 LangGraph 节点签名的 callable.
    """

    def approval_gate(state: AgentState):
        """拦截需要授权的 tool_calls，生成拒绝消息或更新授权集合."""
        last_msg = state["messages"][-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return {"messages": [], "globally_approved_tools": []}

        globally_approved: Set[str] = set(state.get("globally_approved_tools", []))
        rejected_messages: List[ToolMessage] = []

        for tc in last_msg.tool_calls:
            name = tc.get("name", "")
            tool_id = tc.get("id", "")

            try:
                tool = tool_registry.get(name)
            except KeyError:
                continue

            if not getattr(tool, "requires_approval", False):
                continue

            if name in globally_approved:
                continue

            if not interactive:
                # 非交互模式：直接拒绝
                rejected_messages.append(
                    ToolMessage(
                        content=f"[系统] 工具 '{name}' 需要用户授权，但当前处于非交互模式。调用已拒绝。",
                        tool_call_id=tool_id,
                    )
                )
                logger.warning(f"[Approval] 非交互模式下拒绝 '{name}'")
                continue

            # 交互模式：询问用户
            args = tc.get("args", {})
            if hasattr(args, "dict"):
                args = args.dict()
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())

            print(f"\n[Approval Request] {name}({args_str})")
            while True:
                try:
                    choice = input("[y]es / [n]o / [a]ll / [q]uit: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\n[系统] 用户取消输入，拒绝本次调用。")
                    choice = "n"

                if choice in ("y", "yes"):
                    # 单次同意：不加入全局集合
                    logger.info(f"[Approval] 用户单次同意 '{name}'")
                    break
                elif choice in ("n", "no"):
                    rejected_messages.append(
                        ToolMessage(
                            content=f"[系统] 用户拒绝了 '{name}' 的调用。请尝试其他方式或询问用户。",
                            tool_call_id=tool_id,
                        )
                    )
                    logger.info(f"[Approval] 用户拒绝 '{name}'")
                    break
                elif choice in ("a", "all"):
                    globally_approved.add(name)
                    logger.info(f"[Approval] 用户全局授权 '{name}'")
                    break
                elif choice in ("q", "quit"):
                    print("\n[系统] 用户退出会话。")
                    sys.exit(0)
                else:
                    print("请输入 y / n / a / q")

        return {
            "messages": rejected_messages,
            "globally_approved_tools": list(globally_approved),
        }

    return approval_gate
