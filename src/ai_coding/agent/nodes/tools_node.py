"""Tools 节点工厂.

负责执行工具调用，并根据工具类型更新文件快照.
"""

from typing import Dict, List

from langchain_core.messages import AIMessage, ToolMessage

from ai_coding.agent.state import AgentState
from ai_coding.logger import get_logger
from ai_coding.tools.base import ToolRegistry
from ai_coding.tools.todo_tool import TodoTool

logger = get_logger(__name__)


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
    """从 write_file/str_replace_file 成功消息中提取路径."""
    if "文件已写入:" in result:
        return result.split("文件已写入:", 1)[1].split("(")[0].strip()
    if "文件已编辑:" in result:
        return result.split("文件已编辑:", 1)[1].strip()
    return ""


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

        # 同步 state 中的 todos 到 TodoTool 实例（确保跨轮次一致性）
        todo_tool = tool_registry.get("set_todo")
        if isinstance(todo_tool, TodoTool):
            todo_tool.sync(state.get("todos", []))

        tool_messages = []
        file_snapshots: Dict[str, str] = {}
        updated_todos: List[dict] = []

        for tc in last_msg.tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {})
            tool_id = tc.get("id", "")

            # 确保 args 是 dict
            if hasattr(args, "dict"):
                args = args.dict()
            elif not isinstance(args, dict):
                args = dict(args)

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

            elif name in ("str_replace_file", "insert_after_line"):
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

        return {"messages": tool_messages, "file_snapshots": file_snapshots, "todos": updated_todos}

    return tools_node
