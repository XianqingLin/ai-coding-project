"""MCP Tool 适配器.

将外部 MCP Server 的工具元数据包装为项目内部 Tool 接口,
使 LangGraph Agent 可以像调用内置工具一样调用 MCP 工具.
"""

from __future__ import annotations

from typing import Any, List

from mcp.types import Tool as MCPTool

from ai_coding.mcp.client import MCPClient, format_tool_result
from ai_coding.tools.base import Tool, ToolParameter


class MCPToolAdapter(Tool):
    """内部 Tool 到 MCP Tool 的适配器."""

    def __init__(
        self,
        server_name: str,
        mcp_tool: MCPTool,
        client: MCPClient,
        prefix: str = "mcp",
    ) -> None:
        """初始化适配器.

        Args:
            server_name: MCP Server 名称, 用于隔离工具名.
            mcp_tool: MCP 工具元数据.
            client: 已连接的 MCPClient 实例.
            prefix: 工具名前缀.
        """
        self.server_name = server_name
        self.mcp_tool = mcp_tool
        self.client = client
        self.prefix = prefix

        self.name = f"{prefix}_{server_name}_{mcp_tool.name}"
        self.description = mcp_tool.description or f"MCP tool: {mcp_tool.name}"
        self.requires_approval = False

    @property
    def parameters(self) -> List[ToolParameter]:
        """把 MCP inputSchema 转换为内部 ToolParameter 列表."""
        schema = self.mcp_tool.inputSchema or {}
        properties = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])

        params: List[ToolParameter] = []
        for prop_name, prop_schema in properties.items():
            if not isinstance(prop_schema, dict):
                prop_schema = {}
            param_type = self._map_type(prop_schema.get("type", "string"))
            description = prop_schema.get("description", f"参数 {prop_name}")
            enum = prop_schema.get("enum")
            params.append(
                ToolParameter(
                    name=prop_name,
                    param_type=param_type,
                    description=description,
                    required=prop_name in required,
                    enum=list(enum) if enum is not None else None,
                )
            )
        return params

    @staticmethod
    def _map_type(json_type: Any) -> str:
        """将 JSON Schema 类型映射为内部类型标识."""
        if isinstance(json_type, list):
            # 例如 ["string", "null"], 取第一个非 null 类型
            for t in json_type:
                if t != "null":
                    return MCPToolAdapter._map_type(t)
            return "string"
        type_map = {
            "string": "string",
            "integer": "integer",
            "boolean": "boolean",
            "number": "number",
            "array": "array",
            "object": "object",
        }
        return type_map.get(str(json_type), "string")

    def execute(self, **kwargs: Any) -> str:
        """调用 MCP Server 上的工具并返回文本结果."""
        result = self.client.call_tool(self.mcp_tool.name, dict(kwargs))
        return format_tool_result(result)
