"""工具系统包.

提供 AI Coding 助手可用的所有工具.
"""

from ai_coding.tools.base import Tool, ToolParameter, ToolCall, ToolRegistry
from ai_coding.tools.file_tools import (
    ReadFileTool,
    WriteFileTool,
    EditFile,
    ListDirTool,
    GrepTool,
    GlobTool,
)
from ai_coding.tools.collaboration_tools import AgentTool, AskUserQuestionTool
from ai_coding.tools.plan_tools import EnterPlanModeTool, ExitPlanModeTool
from ai_coding.tools.shell_tools import ExecuteCommandTool
from ai_coding.tools.task_tools import TaskListTool, TaskOutputTool, TaskStopTool
from ai_coding.tools.todo_tool import TodoTool


def create_default_tools() -> list:
    """创建一组全新的默认工具实例，避免跨会话/跨 Agent 共享可变状态."""
    execute_tool = ExecuteCommandTool()
    return [
        ReadFileTool(),
        WriteFileTool(),
        EditFile(),
        ListDirTool(),
        GrepTool(),
        GlobTool(),
        EnterPlanModeTool(),
        ExitPlanModeTool(),
        AskUserQuestionTool(),
        AgentTool(),
        execute_tool,
        TaskListTool(task_manager=execute_tool),
        TaskOutputTool(task_manager=execute_tool),
        TaskStopTool(task_manager=execute_tool),
        TodoTool(),
    ]


# 保留旧常量名，指向一组新建实例，便于兼容旧代码；
# 推荐在新代码中调用 create_default_tools() 以获得隔离实例。
DEFAULT_TOOLS = create_default_tools()

__all__ = [
    "Tool",
    "ToolParameter",
    "ToolCall",
    "ToolRegistry",
    "ReadFileTool",
    "WriteFileTool",
    "EditFile",
    "ListDirTool",
    "GrepTool",
    "GlobTool",
    "ExecuteCommandTool",
    "TaskListTool",
    "TaskOutputTool",
    "TaskStopTool",
    "TodoTool",
    "EnterPlanModeTool",
    "ExitPlanModeTool",
    "AskUserQuestionTool",
    "AgentTool",
    "create_default_tools",
    "DEFAULT_TOOLS",
]
