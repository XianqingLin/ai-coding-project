"""工具系统基类.

提供 Tool 抽象基类和 ToolRegistry 工具注册表.
支持转换为 LangChain StructuredTool.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


@dataclass
class ToolParameter:
    """工具参数定义."""

    name: str
    param_type: str  # string, integer, boolean, etc.
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Any = None


@dataclass
class ToolResult:
    """统一的工具执行结果.

    所有工具执行后必须返回 ToolResult，调用方通过 success/error_code 做程序化
    判断，data 用于承载返回给 LLM 的文本内容，便于日志记录和错误重试。
    """

    success: bool
    data: str
    error_code: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    # 用于自动推断字符串结果的状态（保留给历史测试/序列化使用）
    _ERROR_PREFIXES = ("[错误]", "[超时]", "[系统]")

    @classmethod
    def from_string(cls, text: str) -> "ToolResult":
        """将遗留的字符串结果转换为 ToolResult.

        如果字符串以常见错误前缀开头，则视为失败；否则视为成功。
        保留给历史测试或序列化使用，ToolRegistry.execute 不再调用它。
        """
        success = not any(text.startswith(p) for p in cls._ERROR_PREFIXES)
        return cls(success=success, data=text)

    @classmethod
    def ok(cls, data: str, metadata: Optional[Dict[str, Any]] = None) -> "ToolResult":
        """构造成功结果."""
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def fail(cls, data: str, error_code: Optional[str] = None) -> "ToolResult":
        """构造失败结果."""
        return cls(success=False, data=data, error_code=error_code)


class Tool(ABC):
    """工具抽象基类.

    所有具体工具都必须继承此类, 并实现 name, description, parameters 和 execute 方法.
    支持转换为 LangChain StructuredTool.

    Example:
        >>> class ReadFileTool(Tool):
        ...     name = "read_file"
        ...     description = "读取文件内容"
        ...
        ...     @property
        ...     def parameters(self) -> List[ToolParameter]:
        ...         return [ToolParameter("path", "string", "文件路径")]
        ...
        ...     def execute(self, path: str) -> str:
        ...         with open(path, "r") as f:
        ...             return f.read()

    """

    name: str = ""
    description: str = ""
    requires_approval: bool = False
    work_dir: str = ""

    def set_work_dir(self, work_dir: str) -> None:
        """设置工具允许操作的工作目录根路径."""
        self.work_dir = work_dir or ""

    def _resolve_path(self, path: str, must_exist: bool = False) -> "Path":
        """解析并校验路径位于工作目录沙箱内."""
        from ai_coding.tools.safety import resolve_workdir_path

        return resolve_workdir_path(path, self.work_dir, must_exist=must_exist)

    @property
    @abstractmethod
    def parameters(self) -> List[ToolParameter]:
        """返回工具的参数定义列表."""
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> "ToolResult":
        """执行工具逻辑.

        Args:
            **kwargs: 由 LLM 提供的参数.

        Returns:
            工具执行结果（ToolResult）。

        """
        ...

    def get_schema(self) -> dict:
        """生成符合 OpenAI Function Calling 格式的 Schema.

        Returns:
            JSON Schema 格式的工具定义.

        """
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param in self.parameters:
            prop: Dict[str, Any] = {
                "type": param.param_type,
                "description": param.description,
            }
            if param.enum is not None:
                prop["enum"] = param.enum
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def validate_args(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """验证并补全参数.

        检查必填参数是否缺失, 并为可选参数填充默认值.

        Args:
            arguments: LLM 提供的参数.

        Returns:
            验证后的参数.

        Raises:
            ValueError: 必填参数缺失时.

        """
        validated = dict(arguments)

        for param in self.parameters:
            if param.name not in validated:
                if param.required:
                    raise ValueError(f"工具 '{self.name}' 缺少必填参数: '{param.name}'")
                if param.default is not None:
                    validated[param.name] = param.default

        return validated

    def to_langchain_tool(self) -> "BaseTool":
        """转换为 LangChain StructuredTool.

        Returns:
            LangChain 格式的工具实例.

        """
        from langchain_core.tools import StructuredTool
        from pydantic import Field, create_model

        # 动态创建 Pydantic 参数模型
        fields = {}
        for param in self.parameters:
            # 类型映射
            type_map = {
                "string": str,
                "integer": int,
                "boolean": bool,
                "number": float,
                "array": list,
                "object": dict,
            }
            py_type = type_map.get(param.param_type, str)

            if param.required:
                fields[param.name] = (py_type, Field(description=param.description))
            else:
                default = param.default if param.default is not None else None
                fields[param.name] = (
                    py_type,
                    Field(default=default, description=param.description),
                )

        if fields:
            ArgsSchema = create_model(
                f"{self.name.title()}Args", **fields
            )  # type: ignore[call-overload]
        else:
            ArgsSchema = create_model(f"{self.name.title()}Args")

        return StructuredTool.from_function(
            func=self._execute_as_string,
            name=self.name,
            description=self.description,
            args_schema=ArgsSchema,
        )

    def _execute_as_string(self, **kwargs: Any) -> str:
        """供 LangChain 调用的兼容层，保证始终返回字符串."""
        return self.execute(**kwargs).data


class ToolRegistry:
    """工具注册表.

    管理所有可用工具, 负责工具注册、查询和执行.

    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册一个工具.

        Args:
            tool: 要注册的工具实例.

        Raises:
            ValueError: 工具名已存在时.

        """
        if tool.name in self._tools:
            raise ValueError(f"工具 '{tool.name}' 已注册")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """根据名称获取工具.

        Args:
            name: 工具名称.

        Returns:
            工具实例.

        Raises:
            KeyError: 工具不存在时.

        """
        if name not in self._tools:
            raise KeyError(f"未知工具: '{name}'. 可用工具: {self.list_tools()}")
        return self._tools[name]

    def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """执行指定工具并返回统一的 ToolResult.

        Args:
            name: 工具名称.
            arguments: 工具参数.

        Returns:
            工具执行结果（结构化）.

        Raises:
            TypeError: 工具没有返回 ToolResult 时.

        """
        tool = self.get(name)
        validated_args = tool.validate_args(arguments)
        result = tool.execute(**validated_args)
        # 所有工具现在都应返回 ToolResult
        if not isinstance(result, ToolResult):
            raise TypeError(
                f"工具 '{name}' 必须返回 ToolResult，实际返回 {type(result)!r}"
            )
        return result

    def list_tools(self) -> List[str]:
        """列出所有已注册的工具名称.

        Returns:
            工具名称列表.

        """
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
