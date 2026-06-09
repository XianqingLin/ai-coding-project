"""工具系统包.

提供 AI Coding 助手可用的所有工具.
"""

from ai_coding.tools.base import Tool, ToolParameter, ToolCall, ToolRegistry
from ai_coding.tools.file_tools import (
    ReadFileTool,
    WriteFileTool,
    StrReplaceFileTool,
    InsertAfterLineTool,
    ListDirTool,
)
from ai_coding.tools.shell_tools import ExecuteCommandTool
from ai_coding.tools.grep_tool import GrepTool
from ai_coding.tools.todo_tool import TodoTool

# 默认工具集
DEFAULT_TOOLS = [
    ReadFileTool(),
    WriteFileTool(),
    StrReplaceFileTool(),
    InsertAfterLineTool(),

    ListDirTool(),
    ExecuteCommandTool(),
    GrepTool(),
    TodoTool(),
]

__all__ = [
    "Tool",
    "ToolParameter",
    "ToolCall",
    "ToolRegistry",
    "ReadFileTool",
    "WriteFileTool",
    "StrReplaceFileTool",
    "InsertAfterLineTool",

    "ListDirTool",
    "ExecuteCommandTool",
    "GrepTool",
    "TodoTool",
    "DEFAULT_TOOLS",
]
