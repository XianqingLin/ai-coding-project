"""MCP (Model Context Protocol) 集成模块.

提供 MCP Client 与示例 Server, 用于将外部 MCP Server 的工具
动态接入 AI Coding Agent.
"""

from ai_coding.mcp.client import MCPClient

__all__ = ["MCPClient"]
