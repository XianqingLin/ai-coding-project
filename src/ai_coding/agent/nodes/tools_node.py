"""Tools 节点工厂.

负责执行工具调用，并根据工具类型更新文件快照.
"""

import os
import sys
from typing import Dict, List

from langchain_core.messages import AIMessage, ToolMessage

from ai_coding.agent.state import AgentState
from ai_coding.logger import get_logger
from ai_coding.tools.base import ToolRegistry
from ai_coding.tools.collaboration_tools import AskUserQuestionTool
from ai_coding.tools.shell_tools import ExecuteCommandTool
from ai_coding.tools.todo_tool import TodoTool

logger = get_logger(__name__)

# ExitPlanMode 交互保留词
_EXIT_RESERVED = {"a", "approve", "r", "reject", "x", "reject and exit",
                   "v", "revise", "q", "quit"}


def _parse_read_file_result(result: str) -> tuple[str, str]:
    """从 read_file 返回结果中解析 (path, content).

    read_file 返回格式:
        文件: {path}
        ==================================================
           1 | line1
           2 | line2
        ==================================================
        (本段 ...)
    """
    lines = result.split("\n")
    if not lines or not lines[0].startswith("文件:"):
        return "", result

    path = lines[0].replace("文件:", "").strip()

    content_lines = []
    in_content = False
    for line in lines[1:]:
        if line.startswith("=" * 10):
            if in_content:
                break
            in_content = True
            continue
        if in_content:
            # 去掉行号前缀 "   N | "
            if " | " in line:
                line = line.split(" | ", 1)[1]
            content_lines.append(line)

    content = "\n".join(content_lines)
    return path, content


def _extract_path_from_success(result: str) -> str:
    """从 write_file/edit_file 成功消息中提取路径."""
    if "文件已写入:" in result:
        return result.split("文件已写入:", 1)[1].split("(")[0].strip()
    if "文件已编辑:" in result:
        return result.split("文件已编辑:", 1)[1].strip()
    return ""


def _extract_plan_path(result: str) -> str:
    """从 enter_plan_mode 返回结果中提取计划文件路径."""
    for line in result.split("\n"):
        if line.startswith("计划文件路径:"):
            return line.replace("计划文件路径:", "").strip()
    return ""


def _prompt_plan_approval(plan_content: str, options: List[Dict[str, str]]) -> str:
    """交互式呈现计划并等待用户选择.

    Returns:
        "approve", "reject", "reject_and_exit", "revise", 或某个 option label.
    """
    print("\n[Plan Approval]")
    print("=" * 50)
    print(plan_content if plan_content.strip() else "(计划文件为空)")
    print("=" * 50)

    print("\n[选项]")
    print("  a) Approve")
    print("  r) Reject")
    print("  x) Reject and Exit")
    print("  v) Revise")
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt['label']} - {opt.get('description', '')}")

    while True:
        try:
            choice = input("\n请选择 [a/r/x/v" + "".join(str(i + 1) for i in range(len(options))) + "/q]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[系统] 用户取消输入，默认拒绝并退出 Plan 模式。")
            return "reject_and_exit"

        cl = choice.lower()
        if cl in ("q", "quit"):
            print("\n[系统] 用户退出会话。")
            sys.exit(0)
        if cl in ("a", "approve"):
            return "approve"
        if cl in ("r", "reject"):
            return "reject"
        if cl in ("x", "reject and exit"):
            return "reject_and_exit"
        if cl in ("v", "revise"):
            return "revise"

        # 检查是否选了某个 option
        try:
            idx = int(cl) - 1
            if 0 <= idx < len(options):
                return options[idx]["label"]
        except ValueError:
            pass

        print("无效选项，请重新输入。")


def create_tools_node(tool_registry: ToolRegistry):
    """创建 Tools 节点函数.

    Args:
        tool_registry: 工具注册表，用于查找和执行工具.

    Returns:
        符合 LangGraph 节点签名的 callable.
    """

    def tools_node(state: AgentState):
        """执行工具调用并更新文件快照和任务列表."""
        last_msg = state["messages"][-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return {"messages": [], "file_snapshots": {}}

        # 从 messages 中提取已被拒绝的 tool_call_id（由 approval_gate 生成）
        rejected_ids = set()
        for msg in state["messages"]:
            if isinstance(msg, ToolMessage) and "[系统] 用户拒绝了" in msg.content:
                rejected_ids.add(msg.tool_call_id)

        # 同步 state 中的 todos 到 TodoTool 实例（确保跨轮次一致性）
        try:
            todo_tool = tool_registry.get("set_todo")
        except KeyError:
            todo_tool = None
        if isinstance(todo_tool, TodoTool):
            todo_tool.sync(state.get("todos", []))

        # 同步 state 中的 background_tasks 到 ExecuteCommandTool 实例
        try:
            exec_tool = tool_registry.get("execute_command")
        except KeyError:
            exec_tool = None
        if isinstance(exec_tool, ExecuteCommandTool):
            # 将 state 中的任务信息合并到 exec_tool（保留 _proc/_file 引用）
            for t in state.get("background_tasks", []):
                tid = t.get("task_id")
                if tid and tid in exec_tool._bg_tasks:
                    # 保留内部引用，只更新可序列化字段
                    old = exec_tool._bg_tasks[tid]
                    for k, v in t.items():
                        if not k.startswith("_"):
                            old[k] = v

        tool_messages = []
        file_snapshots: Dict[str, str] = {}
        updated_todos: List[dict] = []
        updated_bg_tasks: List[dict] = []
        plan_mode = bool(state.get("plan_mode", False))
        plan_file_path = str(state.get("plan_file_path", ""))

        for tc in last_msg.tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {})
            tool_id = tc.get("id", "")

            # 跳过已被拒绝的调用
            if tool_id in rejected_ids:
                logger.info(f"[ToolsNode] 跳过已拒绝的调用: {name} ({tool_id})")
                continue

            # 确保 args 是 dict
            if hasattr(args, "dict"):
                args = args.dict()
            elif not isinstance(args, dict):
                args = dict(args)

            # ---------- Plan 模式约束 ----------
            if plan_mode and name in ("write_file", "edit_file"):
                target = args.get("path", "")
                if target != plan_file_path:
                    msg = (
                        f"[错误] Plan 模式下只能修改计划文件 '{plan_file_path}'，"
                        f"不允许写入 '{target}'"
                    )
                    tool_messages.append(ToolMessage(content=msg, tool_call_id=tool_id))
                    logger.warning(f"[PlanMode] 拦截 {name} 到非计划文件: {target}")
                    continue

            if plan_mode and name == "task_stop":
                msg = "[错误] Plan 模式下不能使用 task_stop 工具"
                tool_messages.append(ToolMessage(content=msg, tool_call_id=tool_id))
                logger.warning("[PlanMode] 拦截 task_stop")
                continue

            # ---------- enter_plan_mode 特殊处理 ----------
            if name == "enter_plan_mode":
                result = tool_registry.execute(name, args)
                tool_messages.append(ToolMessage(content=result, tool_call_id=tool_id))
                extracted = _extract_plan_path(result)
                if extracted:
                    plan_mode = True
                    plan_file_path = extracted
                    logger.info(f"[PlanMode] 已进入 Plan 模式，计划文件: {plan_file_path}")
                continue

            # ---------- exit_plan_mode 特殊处理 ----------
            if name == "exit_plan_mode":
                if not plan_mode:
                    msg = "[错误] 当前不在 Plan 模式中"
                    tool_messages.append(ToolMessage(content=msg, tool_call_id=tool_id))
                    continue

                # 读取计划文件内容
                plan_content = ""
                if plan_file_path and os.path.exists(plan_file_path):
                    try:
                        with open(plan_file_path, "r", encoding="utf-8") as f:
                            plan_content = f.read()
                    except Exception as e:
                        plan_content = f"(读取计划文件失败: {e})"

                # 解析 options
                raw_options = args.get("options", [])
                options: List[Dict[str, str]] = []
                if isinstance(raw_options, list):
                    for opt in raw_options:
                        if isinstance(opt, dict) and "label" in opt:
                            options.append({
                                "label": str(opt["label"]),
                                "description": str(opt.get("description", "")),
                            })

                choice = _prompt_plan_approval(plan_content, options)

                if choice == "approve":
                    plan_mode = False
                    plan_file_path = ""
                    msg = "[成功] 计划已批准，已退出 Plan 模式。现在可以执行计划中的操作。"
                    logger.info("[PlanMode] 用户批准计划，已退出 Plan 模式")
                elif choice == "reject":
                    msg = "[系统] 用户拒绝了计划。请根据反馈修改计划后重试。"
                    logger.info("[PlanMode] 用户拒绝计划，保持 Plan 模式")
                elif choice == "reject_and_exit":
                    plan_mode = False
                    plan_file_path = ""
                    msg = "[系统] 用户拒绝了计划并退出 Plan 模式。"
                    logger.info("[PlanMode] 用户拒绝并退出 Plan 模式")
                elif choice == "revise":
                    msg = "[系统] 用户要求修改计划。请根据反馈修改计划文件后重试。"
                    logger.info("[PlanMode] 用户要求修改计划，保持 Plan 模式")
                else:
                    # 用户选择了某个 option
                    plan_mode = False
                    plan_file_path = ""
                    msg = f"[成功] 用户选择了方案 '{choice}'，已退出 Plan 模式。请按该方案执行。"
                    logger.info(f"[PlanMode] 用户选择方案 '{choice}'，已退出 Plan 模式")

                tool_messages.append(ToolMessage(content=msg, tool_call_id=tool_id))
                continue

            # ---------- ask_user_question 特殊处理 ----------
            if name == "ask_user_question":
                question = args.get("question", "")
                options = args.get("options", [])
                multi_select = args.get("multi_select", False)
                ask_tool = tool_registry.get("ask_user_question")
                if isinstance(ask_tool, AskUserQuestionTool):
                    result = ask_tool.execute(
                        question=question,
                        options=options,
                        multi_select=multi_select,
                    )
                    tool_messages.append(ToolMessage(content=result, tool_call_id=tool_id))
                continue

            # ---------- 常规工具执行 ----------
            result = tool_registry.execute(name, args)
            tool_messages.append(ToolMessage(content=result, tool_call_id=tool_id))

            # 根据工具类型更新文件快照
            if name == "read_file":
                path = args.get("path", "")
                if path:
                    _, content = _parse_read_file_result(result)
                    file_snapshots[path] = content
                    logger.debug(f"[FileSnapshot] 读取更新: {path}")

            elif name == "write_file":
                path = args.get("path", "")
                if path:
                    # write_file 的 content 就是最新完整内容
                    file_snapshots[path] = args.get("content", "")
                    logger.debug(f"[FileSnapshot] 写入更新: {path}")

            elif name == "edit_file":
                path = args.get("path", "")
                if path and result.startswith("[成功]"):
                    try:
                        read_result = tool_registry.execute("read_file", {"path": path})
                        _, content = _parse_read_file_result(read_result)
                        file_snapshots[path] = content
                        logger.debug(f"[FileSnapshot] 编辑后刷新: {path}")
                    except Exception as e:
                        logger.warning(f"[FileSnapshot] 编辑后刷新失败 {path}: {e}")

        # 收集更新后的 todos（如果有 set_todo 调用或需要同步现有状态）
        if isinstance(todo_tool, TodoTool):
            updated_todos = todo_tool.todos

        # 收集更新后的后台任务状态
        if isinstance(exec_tool, ExecuteCommandTool):
            updated_bg_tasks = exec_tool.background_tasks()

        # 收集子 Agent 状态（透传 state 中的 sub_agents）
        updated_sub_agents = list(state.get("sub_agents", []))

        return {
            "messages": tool_messages,
            "file_snapshots": file_snapshots,
            "todos": updated_todos,
            "background_tasks": updated_bg_tasks,
            "plan_mode": plan_mode,
            "plan_file_path": plan_file_path,
            "sub_agents": updated_sub_agents,
        }

    return tools_node
