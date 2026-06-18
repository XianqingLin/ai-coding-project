"""环境信息收集与渲染模块.

负责收集 Agent 运行时的环境上下文，并将其格式化为可插入系统提示词的文本。
"""

import os
import platform as _platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from ai_coding.config import (
    DEFAULT_LLM_PROVIDER,
    KIMI_MODEL,
    OPENAI_MODEL,
)

_PROVIDER_META: Dict[str, Dict[str, str]] = {
    "kimi": {
        "marketing_name": "Kimi",
        "model_id": KIMI_MODEL,
        "knowledge_cutoff": "2025-01",
    },
    "openai": {
        "marketing_name": "OpenAI GPT",
        "model_id": OPENAI_MODEL,
        "knowledge_cutoff": "2023-12",
    },
    "mock": {
        "marketing_name": "Mock Model",
        "model_id": "mock",
        "knowledge_cutoff": "N/A",
    },
}


@dataclass(frozen=True)
class EnvironmentInfo:
    """Agent 运行时环境信息.

    Attributes:
        cwd: 主工作目录（绝对路径）.
        is_git_repo: 是否为 git 仓库.
        platform: 平台标识.
        shell: 当前 Shell.
        os_version: 操作系统版本.
        marketing_name: 模型营销名.
        model_id: 确切模型 ID.
        knowledge_cutoff: 助手知识截止日期.
        date: 当前日期（YYYY-MM-DD）.
    """

    cwd: str
    is_git_repo: bool
    platform: str
    shell: str
    os_version: str
    marketing_name: str
    model_id: str
    knowledge_cutoff: str
    date: str

    def to_dict(self) -> Dict[str, str]:
        """返回模板渲染所需的字段字典，用于格式化 default_system_prompt.txt."""
        return {
            "cwd": self.cwd,
            "date": self.date,
        }

    def render(self) -> str:
        """渲染为系统提示词中的环境信息文本."""
        git_status = "是" if self.is_git_repo else "否"
        return (
            "# 环境\n"
            "你在以下环境中被调用：\n"
            f" - 主工作目录：{self.cwd}\n"
            f" - 是否为 git 仓库：{git_status}\n"
            f" - 平台：{self.platform}\n"
            f" - Shell：{self.shell}\n"
            f" - 操作系统版本：{self.os_version}\n"
            f" - 你由名为 {self.marketing_name} 的模型驱动。"
            f"确切的模型 ID 是 {self.model_id}。\n"
            f" - 助手知识截止日期为 {self.knowledge_cutoff}。\n"
        )


def _detect_shell() -> str:
    """检测当前使用的 Shell."""
    if sys.platform == "win32":
        return os.environ.get("COMSPEC", "cmd.exe")
    return os.environ.get("SHELL", "/bin/sh")


def _is_git_repo(path: Path) -> bool:
    """检测指定路径是否位于 git 仓库内.

    优先通过目录结构判断，避免子进程开销。
    """
    try:
        resolved = path.resolve()
        for parent in [resolved, *resolved.parents]:
            if (parent / ".git").exists():
                return True
    except OSError:
        pass
    return False


def collect_environment_info(
    work_dir: str,
    provider: Optional[str] = None,
) -> EnvironmentInfo:
    """收集指定工作目录的环境信息.

    Args:
        work_dir: 主工作目录.
        provider: LLM 提供商名称，用于决定模型元数据.

    Returns:
        环境信息对象.
    """
    provider = (provider or DEFAULT_LLM_PROVIDER).lower()
    meta = _PROVIDER_META.get(provider, _PROVIDER_META["mock"])

    cwd_path = Path(work_dir).expanduser().resolve()

    return EnvironmentInfo(
        cwd=str(cwd_path),
        is_git_repo=_is_git_repo(cwd_path),
        platform=_platform.system() or sys.platform,
        shell=_detect_shell(),
        os_version=_platform.platform(),
        marketing_name=meta["marketing_name"],
        model_id=meta["model_id"],
        knowledge_cutoff=meta["knowledge_cutoff"],
        date=datetime.now().strftime("%Y-%m-%d"),
    )
