"""Approval Gate 节点工厂.

在 LLM 输出 tool_calls 之后、tools 节点执行之前插入，
负责拦截需要用户授权的写操作工具，提供交互式确认。

支持两种交互模式：
- 前端驱动：通过 event_loop + on_approval_request 回调等待前端响应
  （用于 VSCode 插件、Web 前端等场景）
- 命令行驱动：直接调用 input() 询问用户
  （用于 CLI/TUI 场景，无 event_loop 时回退）
- 非交互模式：直接拒绝所有未授权调用
"""

from concurrent.futures import Future, TimeoutError
from typing import Any, Callable, Dict, List, Optional, Set

from langchain_core.messages import AIMessage, ToolMessage

from ai_coding.agent.nodes._utils import normalize_tool_args
from ai_coding.agent.state import AgentState
from ai_coding.logger import get_logger
from ai_coding.tools.base import ToolRegistry

logger = get_logger(__name__)

# 前端审批等待超时时间（秒）
APPROVAL_TIMEOUT_SECONDS = 300


def create_approval_gate(
    tool_registry: ToolRegistry,
    interactive: bool = True,
    on_approval_request: Optional[Callable[[Dict[str, Any]], None]] = None,
    register_approval_future: Optional[Callable[[str, Future], None]] = None,
):
    """创建 Approval Gate 节点函数.

    检查 LLM 输出的 tool_calls 中是否有需要用户授权的调用。

    Args:
        tool_registry: 工具注册表，用于查询工具是否需要授权.
        interactive: 是否为交互模式（False 时直接拒绝未授权调用）.
        event_loop: 外部事件循环（保留给 on_approval_request 使用）.
        on_approval_request: approval 请求回调，用于向前端发送请求.
        register_approval_future: 注册创建的 concurrent.futures.Future，按 request_id 索引.

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

        # 先收集所有需要前端审批的请求，再统一等待；避免单 future 覆盖导致多请求丢失
        pending_requests: List[Dict[str, Any]] = []

        for tc in last_msg.tool_calls:
            name = tc.get("name", "")
            tool_call_id = tc.get("id", "")

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
                        tool_call_id=tool_call_id,
                    )
                )
                logger.warning(f"[Approval] 非交互模式下拒绝 '{name}'")
                continue

            args = normalize_tool_args(tc.get("args", {}))
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())

            # 前端驱动模式
            if event_loop is not None and on_approval_request is not None:
                request_id = f"apr_{tool_call_id}" if tool_call_id else f"apr_{id(tc)}"
                request = {
                    "type": "approval_request",
                    "id": request_id,
                    "tool": name,
                    "args": args,
                    "description": f"{name}({args_str})",
                }

                future: Future[str] = Future()
                if register_approval_future is not None:
                    register_approval_future(request_id, future)

                try:
                    on_approval_request(request)
                except Exception as e:
                    logger.error(f"[Approval] 发送审批请求失败: {e}", exc_info=True)
                    rejected_messages.append(
                        ToolMessage(
                            content=f"[系统] 发送 '{name}' 审批请求失败: {e}。调用已拒绝。",
                            tool_call_id=tool_call_id,
                        )
                    )
                    continue

                pending_requests.append({
                    "future": future,
                    "request_id": request_id,
                    "tool": name,
                    "tool_call_id": tool_call_id,
                })
                continue

            # 命令行回退模式
            print(f"\n[Approval Request] {name}({args_str})")
            while True:
                try:
                    choice = input("[y]es / [n]o / [a]ll / [q]uit: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\n[系统] 用户取消输入，拒绝本次调用。")
                    choice = "n"

                if choice in ("y", "yes"):
                    logger.info(f"[Approval] 用户单次同意 '{name}'")
                    break
                elif choice in ("n", "no"):
                    rejected_messages.append(
                        ToolMessage(
                            content=f"[系统] 用户拒绝了 '{name}' 的调用。请尝试其他方式或询问用户。",
                            tool_call_id=tool_call_id,
                        )
                    )
                    logger.info(f"[Approval] 用户拒绝 '{name}'")
                    break
                elif choice in ("a", "all"):
                    globally_approved.add(name)
                    logger.info(f"[Approval] 用户全局授权 '{name}'")
                    break
                elif choice in ("q", "quit"):
                    print("\n[系统] 用户选择退出会话，本次调用已拒绝。")
                    rejected_messages.append(
                        ToolMessage(
                            content=f"[系统] 用户退出会话，'{name}' 调用已拒绝。",
                            tool_call_id=tool_call_id,
                        )
                    )
                    logger.info(f"[Approval] 用户退出会话，拒绝 '{name}'")
                    break
                else:
                    print("请输入 y / n / a / q")

        # 统一等待前端审批响应
        for pending in pending_requests:
            future = pending["future"]
            name = pending["tool"]
            tool_call_id = pending["tool_call_id"]
            choice = "reject"

            try:
                choice = future.result(timeout=APPROVAL_TIMEOUT_SECONDS)
            except TimeoutError:
                logger.warning(f"[Approval] 等待 '{name}' 响应超时，视为拒绝")
                choice = "reject"
            except Exception as e:
                logger.error(f"[Approval] 等待前端响应失败: {e}", exc_info=True)
                choice = "reject"

            if choice in ("approve", "yes", "y"):
                logger.info(f"[Approval] 前端单次同意 '{name}'")
                continue
            elif choice in ("approve_all", "all", "a"):
                globally_approved.add(name)
                logger.info(f"[Approval] 前端全局授权 '{name}'")
                continue
            else:
                rejected_messages.append(
                    ToolMessage(
                        content=f"[系统] 用户拒绝了 '{name}' 的调用。请尝试其他方式或询问用户。",
                        tool_call_id=tool_call_id,
                    )
                )
                logger.info(f"[Approval] 前端拒绝 '{name}'")
                continue

        return {
            "messages": rejected_messages,
            "globally_approved_tools": list(globally_approved),
        }

    return approval_gate
