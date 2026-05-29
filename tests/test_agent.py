"""工具系统测试."""

import pytest

from ai_coding.tools import DEFAULT_TOOLS
from ai_coding.tools.base import Tool, ToolParameter, ToolRegistry


class TestTools:
    """测试工具系统."""

    def test_read_file_tool_exists(self) -> None:
        """测试 read_file 工具存在."""
        tool = next(t for t in DEFAULT_TOOLS if t.name == "read_file")
        assert tool.name == "read_file"
        assert len(tool.parameters) == 1
        assert tool.parameters[0].name == "path"

    def test_list_dir_tool_exists(self) -> None:
        """测试 list_dir 工具存在."""
        tool = next(t for t in DEFAULT_TOOLS if t.name == "list_dir")
        assert tool.name == "list_dir"

    def test_tool_schema(self) -> None:
        """测试工具 Schema 生成."""
        tool = next(t for t in DEFAULT_TOOLS if t.name == "read_file")
        schema = tool.get_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "read_file"
        assert "parameters" in schema["function"]

    def test_tool_registry(self) -> None:
        """测试工具注册表."""
        registry = ToolRegistry()
        registry.register_all(DEFAULT_TOOLS)
        assert len(registry) == len(DEFAULT_TOOLS)
        assert "read_file" in registry
        schemas = registry.get_schemas()
        assert len(schemas) == len(DEFAULT_TOOLS)
