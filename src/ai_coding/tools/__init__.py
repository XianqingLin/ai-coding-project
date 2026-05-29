"""工具系统包.

提供 AI Coding 助手可用的所有工具.
"""

from ai_coding.tools.base import Tool, ToolParameter, ToolCall, ToolRegistry
from ai_coding.tools.file_tools import (
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    ListDirTool,
    ExecuteCommandTool,
)

# 默认工具集
DEFAULT_TOOLS = [
    ReadFileTool(),
    WriteFileTool(),
    EditFileTool(),
    ListDirTool(),
    ExecuteCommandTool(),
]

__all__ = [
    "Tool",
    "ToolParameter",
    "ToolCall",
    "ToolRegistry",
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "ListDirTool",
    "ExecuteCommandTool",
    "DEFAULT_TOOLS",
]
