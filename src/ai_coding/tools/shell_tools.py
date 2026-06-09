"""Shell 命令执行相关工具.

提供在系统 shell 中执行命令的能力，用于运行测试、安装依赖、构建项目等.
"""

import subprocess
from typing import List

from ai_coding.tools.base import Tool, ToolParameter


class ExecuteCommandTool(Tool):
    """执行 shell 命令."""

    name = "execute_command"
    requires_approval = True
    description = (
        "执行 shell 命令. 用于运行测试、安装依赖、构建项目、查看目录结构等操作. "
        "谨慎使用有破坏性的命令."
    )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("command", "string", "要执行的 shell 命令"),
            ToolParameter(
                "timeout",
                "integer",
                "命令超时时间（秒）, 默认 30 秒",
                required=False,
                default=30,
            ),
        ]

    def execute(self, command: str, timeout: int = 30) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            output_parts = []

            if result.stdout:
                output_parts.append(f"[stdout]\n{result.stdout}")

            if result.stderr:
                output_parts.append(f"[stderr]\n{result.stderr}")

            if result.returncode != 0:
                output_parts.append(f"[退出码] {result.returncode}")

            if not output_parts:
                return "[成功] 命令执行完成, 无输出."

            return "\n\n".join(output_parts)

        except subprocess.TimeoutExpired:
            return f"[错误] 命令执行超时（超过 {timeout} 秒）."
        except Exception as e:
            return f"[错误] 命令执行失败: {e}"
