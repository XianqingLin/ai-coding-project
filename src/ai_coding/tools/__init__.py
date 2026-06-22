"""工具系统包.

提供 AI Coding 助手可用的所有工具.
"""

from typing import List

from ai_coding.tools.base import Tool, ToolParameter, ToolRegistry
from ai_coding.tools.collaboration_tools import AgentTool, AskUserQuestionTool
from ai_coding.tools.file_tools import (
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
    SearchReplaceTool,
    WriteFileTool,
)
from ai_coding.tools.git_tools import (
    GitAddTool,
    GitBranchCreateTool,
    GitBranchListTool,
    GitBranchSwitchTool,
    GitCommitTool,
    GitDiffTool,
    GitLogTool,
    GitPushTool,
    GitStatusTool,
)
from ai_coding.tools.plan_tools import EnterPlanModeTool, ExitPlanModeTool
from ai_coding.tools.shell_tools import ExecuteCommandTool
from ai_coding.tools.task_tools import TaskListTool, TaskOutputTool, TaskStopTool
from ai_coding.tools.todo_tool import TodoTool


def create_default_tools() -> List[Tool]:
    """创建一组全新的默认工具实例，避免跨会话/跨 Agent 共享可变状态.

    Returns:
        工具实例列表.
    """
    execute_tool = ExecuteCommandTool()
    tools: List[Tool] = [
        ReadFileTool(),
        WriteFileTool(),
        SearchReplaceTool(),
        ListDirTool(),
        GrepTool(),
        GlobTool(),
        GitStatusTool(),
        GitDiffTool(),
        GitLogTool(),
        GitBranchListTool(),
        GitBranchCreateTool(),
        GitBranchSwitchTool(),
        GitAddTool(),
        GitCommitTool(),
        GitPushTool(),
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
    return tools


__all__ = [
    "Tool",
    "ToolParameter",
    "ToolRegistry",
    "ReadFileTool",
    "WriteFileTool",
    "SearchReplaceTool",
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
    "GitStatusTool",
    "GitDiffTool",
    "GitLogTool",
    "GitBranchListTool",
    "GitBranchCreateTool",
    "GitBranchSwitchTool",
    "GitAddTool",
    "GitCommitTool",
    "GitPushTool",
    "create_default_tools",
]
