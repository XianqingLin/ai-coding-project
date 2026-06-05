"""AI Coding - 一个基于 AI 的代码辅助项目.

本项目是一个控制台交互式 AI 编程助手.
采用 LangGraph ReAct Agent 架构, 支持工具调用.

Example:
    >>> from ai_coding import LangGraphAgent, create_lc_llm, DEFAULT_TOOLS
    >>> agent = LangGraphAgent(llm=create_lc_llm("kimi"), tools=DEFAULT_TOOLS)
    >>> result = agent.run("帮我查看 README.md")
    >>> print(result)

"""

__version__ = "0.1.0"

from ai_coding.langgraph_agent import LangGraphAgent
from ai_coding.llm import create_lc_llm
from ai_coding.tools import DEFAULT_TOOLS
from ai_coding.tools.base import Tool, ToolParameter, ToolCall, ToolRegistry

__all__ = [
    "LangGraphAgent",
    "create_lc_llm",
    "DEFAULT_TOOLS",
    "Tool",
    "ToolParameter",
    "ToolCall",
    "ToolRegistry",
    "__version__",
]
