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
from ai_coding.tools.shell_tools import ExecuteCommandTool
from ai_coding.tools.todo_tool import TodoTool

# 默认工具集
DEFAULT_TOOLS = [
    ReadFileTool(),
    WriteFileTool(),
    EditFile(),
    ListDirTool(),
    GrepTool(),
    GlobTool(),
    ExecuteCommandTool(),
    TodoTool(),
]

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
    "TodoTool",
    "DEFAULT_TOOLS",
]
