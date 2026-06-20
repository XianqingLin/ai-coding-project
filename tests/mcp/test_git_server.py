"""本地 Git MCP Server 测试."""

from __future__ import annotations

from ai_coding.mcp.client import MCPClient, format_tool_result
from ai_coding.tools.mcp_adapter import MCPToolAdapter


def test_git_server_connect_and_list_tools() -> None:
    """连接本地 Git MCP Server 并列出工具."""
    client = MCPClient(
        server_name="local_git",
        command="python",
        args=["-m", "ai_coding.mcp.servers.git_server"],
    )
    assert client.connect() is True

    tools = client.list_tools()
    names = {t.name for t in tools}
    assert "get_git_status" in names
    assert "get_git_log" in names
    assert "get_git_branch" in names

    client.stop()


def test_git_server_call_status() -> None:
    """调用 get_git_status 工具."""
    client = MCPClient(
        server_name="local_git",
        command="python",
        args=["-m", "ai_coding.mcp.servers.git_server"],
    )
    assert client.connect() is True

    result = client.call_tool("get_git_status", {"path": "."})
    text = format_tool_result(result)
    assert isinstance(text, str)

    client.stop()


def test_mcp_tool_adapter() -> None:
    """MCP Tool 能被适配为内部 Tool."""
    client = MCPClient(
        server_name="local_git",
        command="python",
        args=["-m", "ai_coding.mcp.servers.git_server"],
    )
    assert client.connect() is True

    tools = client.list_tools()
    status_tool = next(t for t in tools if t.name == "get_git_status")
    adapter = MCPToolAdapter(
        server_name="local_git",
        mcp_tool=status_tool,
        client=client,
    )

    assert adapter.name == "mcp_local_git_get_git_status"
    assert adapter.description
    params = {p.name: p for p in adapter.parameters}
    assert "path" in params
    assert params["path"].required is True

    result = adapter.execute(path=".")
    assert isinstance(result, str)

    client.stop()
