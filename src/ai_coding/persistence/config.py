"""持久化配置.

仅通过环境变量 AI_CODE_HOME 控制存储根目录.
未设置时默认使用 ~/.ai-coding.
"""

import os
from pathlib import Path


_ENV_VAR = "AI_CODE_HOME"


def get_storage_root() -> Path:
    """获取存储根目录.

    优先读取环境变量 AI_CODE_HOME：
    - 相对路径基于当前工作目录解析
    - 绝对路径直接使用
    未设置则回退到 ~/.ai-coding.
    """
    env = os.getenv(_ENV_VAR, "")
    if env:
        path = Path(os.path.expanduser(env))
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()
    return Path.home() / ".ai-coding"


def get_config() -> dict:
    """获取当前配置字典（保留扩展接口）."""
    return {
        "storage_root": str(get_storage_root()),
    }
