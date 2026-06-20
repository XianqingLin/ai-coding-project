"""System prompt 构建器.

负责将静态模板、动态上下文与行为准则拼接为最终发送给模型的 system prompt。
"""

from dataclasses import dataclass
from importlib.resources import files
from typing import List, Optional

from ai_coding.environment import EnvironmentInfo
from ai_coding.logger import get_logger

logger = get_logger(__name__)

# 主模板：直接内联在 builder 中，不再依赖外部文件
MAIN_TEMPLATE = "你是一个 AI 编程助手, 专门帮助用户进行代码开发任务."


@dataclass
class PromptContext:
    """构建 system prompt 时的动态上下文.

    Attributes:
        environment_info: 环境信息，渲染为 # 环境 段落.
        memory_summary: 记忆摘要文本，渲染为 # 记忆摘要 段落.
        language: 回复语言偏好，例如 "中文" / "English".
    """

    environment_info: Optional[EnvironmentInfo] = None
    memory_summary: Optional[str] = None
    language: Optional[str] = None

    def _render_environment(self) -> str:
        if self.environment_info is None:
            return ""
        return self.environment_info.render()

    def _render_date(self) -> str:
        if self.environment_info is None or not self.environment_info.date:
            return ""
        return f"日期：{self.environment_info.date}"

    def _render_memory(self) -> str:
        if not self.memory_summary:
            return ""
        return f"# 记忆摘要\n\n{self.memory_summary}\n"

    def _render_language(self) -> str:
        if not self.language:
            return ""
        return f"请使用 {self.language} 回答用户。\n"

    def render_dynamic_parts(self) -> List[str]:
        """按顺序渲染所有动态段落."""
        parts: List[str] = []
        for renderer in [
            self._render_environment,
            self._render_date,
            self._render_memory,
            self._render_language,
        ]:
            text = renderer()
            if text:
                parts.append(text)
        return parts


class SystemPromptBuilder:
    """System prompt 构建器.

    职责：
    - 读取静态主模板与行为准则文件
    - 使用 PromptContext 渲染动态内容
    - 按固定顺序拼接最终 system prompt

    拼接顺序：
    主模板 -> 行为准则 -> 动态上下文（环境信息、日期、记忆、语言等）
    """

    DEFAULT_GUIDELINE_FILES: List[str] = [
        "guidelines_tool_usage.txt",
        "guidelines_task_execution.txt",
        "guidelines_system_behavior.txt",
    ]

    def __init__(
        self,
        guideline_files: Optional[List[str]] = None,
    ) -> None:
        """初始化构建器.

        Args:
            guideline_files: 行为准则文件列表，按顺序拼接.
        """
        self.guideline_files = guideline_files or list(self.DEFAULT_GUIDELINE_FILES)

    def _read_resource(self, filename: str) -> str:
        """读取 prompts 包资源文件.

        Raises:
            FileNotFoundError: 当指定的资源文件不存在时.
        """
        path = files("ai_coding.prompts").joinpath(filename)
        if not path.exists():  # type: ignore[attr-defined]
            raise FileNotFoundError(f"Prompt guideline file not found: {filename}")
        return path.read_text(encoding="utf-8")

    def _build_main_section(self) -> str:
        """构建主模板段落."""
        return MAIN_TEMPLATE

    def build(self, context: Optional[PromptContext] = None) -> str:
        """构建完整 system prompt.

        Args:
            context: 动态上下文，为空时仅拼接静态部分.

        Returns:
            完整 system prompt 文本.
        """
        # 1. 主模板
        body = self._build_main_section()

        # 2. 行为准则文件（静态内容）
        guidelines = [self._read_resource(f) for f in self.guideline_files]

        # 3. 动态上下文段落
        dynamic_parts = context.render_dynamic_parts() if context else []

        # 4. 拼接并过滤空内容：静态在前，动态在后
        parts: List[str] = [body]
        parts.extend(guidelines)
        parts.extend(dynamic_parts)
        return "\n\n".join(part.strip() for part in parts if part.strip())
