"""工具系统基类测试."""

from typing import List

import pytest
from langchain_core.tools import StructuredTool

from ai_coding.tools.base import Tool, ToolParameter, ToolRegistry, ToolResult


class _NoParamsTool(Tool):
    name = "no_params"
    description = "无参数工具"

    @property
    def parameters(self) -> List[ToolParameter]:
        return []

    def execute(self) -> ToolResult:
        return ToolResult.ok("done")


class _OptionalParamTool(Tool):
    name = "optional"
    description = "带可选参数"

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("required_param", "string", "必填"),
            ToolParameter(
                "optional_param",
                "string",
                "可选",
                required=False,
                default="default_value",
            ),
        ]

    def execute(self, required_param: str, optional_param: str = "") -> ToolResult:
        return ToolResult.ok(f"{required_param}:{optional_param}")


class _EnumParamTool(Tool):
    name = "enum_tool"
    description = "带枚举"

    @property
    def parameters(self) -> List[ToolParameter]:
        return [ToolParameter("mode", "string", "模式", enum=["fast", "slow"])]

    def execute(self, mode: str) -> ToolResult:
        return ToolResult.ok(mode)


class TestToolSchema:
    def test_get_schema_no_params(self):
        tool = _NoParamsTool()
        schema = tool.get_schema()
        assert schema["function"]["name"] == "no_params"
        assert schema["function"]["parameters"]["properties"] == {}
        assert schema["function"]["parameters"]["required"] == []

    def test_get_schema_optional_param(self):
        tool = _OptionalParamTool()
        schema = tool.get_schema()
        props = schema["function"]["parameters"]["properties"]
        required = schema["function"]["parameters"]["required"]

        assert "required_param" in props
        assert "optional_param" in props
        assert "required_param" in required
        assert "optional_param" not in required

    def test_get_schema_enum(self):
        tool = _EnumParamTool()
        schema = tool.get_schema()
        assert schema["function"]["parameters"]["properties"]["mode"]["enum"] == [
            "fast",
            "slow",
        ]


class TestToolValidateArgs:
    def test_validates_required_param(self):
        tool = _OptionalParamTool()
        with pytest.raises(ValueError, match="缺少必填参数"):
            tool.validate_args({})

    def test_fills_optional_default(self):
        tool = _OptionalParamTool()
        validated = tool.validate_args({"required_param": "x"})
        assert validated["optional_param"] == "default_value"

    def test_keeps_provided_optional(self):
        tool = _OptionalParamTool()
        validated = tool.validate_args({"required_param": "x", "optional_param": "y"})
        assert validated["optional_param"] == "y"


class TestToolToLangchain:
    def test_converts_to_structured_tool(self):
        tool = _OptionalParamTool()
        lc_tool = tool.to_langchain_tool()
        assert isinstance(lc_tool, StructuredTool)
        assert lc_tool.name == "optional"

    def test_converts_no_params_tool(self):
        tool = _NoParamsTool()
        lc_tool = tool.to_langchain_tool()
        assert isinstance(lc_tool, StructuredTool)


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = _NoParamsTool()
        registry.register(tool)
        assert registry.get("no_params") is tool

    def test_register_duplicate_raises(self):
        registry = ToolRegistry()
        registry.register(_NoParamsTool())
        with pytest.raises(ValueError, match="已注册"):
            registry.register(_NoParamsTool())

    def test_get_unknown_raises(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="未知工具"):
            registry.get("unknown")

    def test_execute_returns_tool_result(self):
        registry = ToolRegistry()
        registry.register(_OptionalParamTool())
        result = registry.execute("optional", {"required_param": "x"})
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.data == "x:default_value"

    def test_execute_wraps_string_error_as_failure(self):
        class _FailTool(Tool):
            name = "fail"
            description = "总是失败"

            @property
            def parameters(self) -> List[ToolParameter]:
                return []

            def execute(self) -> ToolResult:
                return ToolResult.fail("[错误] 出错了")

        registry = ToolRegistry()
        registry.register(_FailTool())
        result = registry.execute("fail", {})
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "[错误] 出错了" in result.data

    def test_contains_and_len(self):
        registry = ToolRegistry()
        registry.register(_NoParamsTool())
        assert "no_params" in registry
        assert "unknown" not in registry
        assert len(registry) == 1
