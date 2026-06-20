"""MCP 工具加载器测试."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Generator

import pytest

from ai_coding.tools.mcp_loader import MCPLoader


@pytest.fixture
def temp_config() -> Generator[str, None, None]:
    """提供一个指向本地 Git Server 的临时配置文件."""
    prev = os.environ.get("MCP_ENABLED")
    os.environ["MCP_ENABLED"] = "true"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(
            {
                "mcpServers": {
                    "local_git": {
                        "command": "python",
                        "args": ["-m", "ai_coding.mcp.servers.git_server"],
                    }
                }
            },
            f,
        )
        path = f.name
    yield path
    os.unlink(path)
    if prev is None:
        os.environ.pop("MCP_ENABLED", None)
    else:
        os.environ["MCP_ENABLED"] = prev


def test_mcp_loader_loads_tools(temp_config: str) -> None:
    """MCPLoader 能读取配置并加载 MCP 工具."""
    loader = MCPLoader(config_path=temp_config)
    tools, clients = loader.load()
    try:
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert "mcp_local_git_get_git_status" in names
        assert "mcp_local_git_get_git_log" in names
        assert "mcp_local_git_get_git_branch" in names
    finally:
        loader.disconnect_all()


def test_mcp_loader_disabled() -> None:
    """MCP_ENABLED=false 时返回空."""
    os.environ["MCP_ENABLED"] = "false"
    loader = MCPLoader(config_path="nonexistent.json")
    tools, clients = loader.load()
    assert tools == []
    assert clients == []
