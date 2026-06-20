"""MCP Tool 适配器测试."""

from __future__ import annotations

from ai_coding.tools.mcp_adapter import MCPToolAdapter


def make_mcp_tool(name: str = "echo", schema: dict | None = None) -> object:
    """构造一个类似 MCP Tool 的简单对象."""

    class FakeTool:
        def __init__(self) -> None:
            self.name = name
            self.description = "fake tool"
            self.inputSchema = schema or {"type": "object", "properties": {}}

    return FakeTool()


def test_map_type_simple() -> None:
    """JSON Schema 基本类型映射正确."""
    assert MCPToolAdapter._map_type("string") == "string"
    assert MCPToolAdapter._map_type("integer") == "integer"
    assert MCPToolAdapter._map_type("boolean") == "boolean"
    assert MCPToolAdapter._map_type("number") == "number"
    assert MCPToolAdapter._map_type("array") == "array"
    assert MCPToolAdapter._map_type("object") == "object"


def test_map_type_null_union() -> None:
    """[string, null] 类型映射为 string."""
    assert MCPToolAdapter._map_type(["string", "null"]) == "string"


def test_map_type_unknown_defaults_to_string() -> None:
    """未知类型默认映射为 string."""
    assert MCPToolAdapter._map_type("foo") == "string"


def test_parameters_conversion() -> None:
    """MCP inputSchema 正确转换为 ToolParameter."""
    schema = {
        "type": "object",
        "properties": {
            "msg": {"type": "string", "description": "消息内容"},
            "count": {"type": "integer", "description": "次数"},
            "verbose": {"type": "boolean", "description": "是否详细"},
        },
        "required": ["msg"],
    }
    mcp_tool = make_mcp_tool("echo", schema)
    adapter = MCPToolAdapter(
        server_name="test",
        mcp_tool=mcp_tool,  # type: ignore[arg-type]
        client=None,  # type: ignore[arg-type]
    )

    params = {p.name: p for p in adapter.parameters}
    assert "msg" in params
    assert params["msg"].param_type == "string"
    assert params["msg"].required is True
    assert params["count"].param_type == "integer"
    assert params["count"].required is False
    assert params["verbose"].param_type == "boolean"


def test_enum_parameter() -> None:
    """enum 参数被正确保留."""
    schema = {
        "type": "object",
        "properties": {
            "level": {
                "type": "string",
                "description": "级别",
                "enum": ["low", "medium", "high"],
            }
        },
        "required": ["level"],
    }
    mcp_tool = make_mcp_tool("set_level", schema)
    adapter = MCPToolAdapter(
        server_name="test",
        mcp_tool=mcp_tool,  # type: ignore[arg-type]
        client=None,  # type: ignore[arg-type]
    )

    param = adapter.parameters[0]
    assert param.name == "level"
    assert param.enum == ["low", "medium", "high"]
