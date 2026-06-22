"""配置管理模块.

负责加载环境变量和项目配置.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


def load_env() -> None:
    """加载 .env 文件中的环境变量."""
    # 从项目根目录加载 .env
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)


def get_env(key: str, default: str = "") -> str:
    """获取环境变量值.

    Args:
        key: 环境变量名.
        default: 默认值.

    Returns:
        环境变量值, 未设置则返回默认值.

    """
    return os.getenv(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """获取布尔类型的环境变量.

    Args:
        key: 环境变量名.
        default: 默认值.

    Returns:
        True 当值为 'true', '1', 'yes', 'on'（不区分大小写）.

    """
    value = os.getenv(key, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    return default


# 加载环境变量（模块导入时自动执行）
load_env()

# 常用配置项
KIMI_API_KEY = get_env("KIMI_API_KEY")
KIMI_BASE_URL = get_env("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
KIMI_MODEL = get_env("KIMI_MODEL", "kimi-k2.6")

OPENAI_API_KEY = get_env("OPENAI_API_KEY")
OPENAI_BASE_URL = get_env("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = get_env("OPENAI_MODEL", "gpt-4")

DEFAULT_LLM_PROVIDER = get_env("DEFAULT_LLM_PROVIDER", "kimi")
LOG_LEVEL = get_env("LOG_LEVEL", "INFO")

# Mock LLM 配置
MOCK_INTERACTIVE = get_env_bool("MOCK_INTERACTIVE", False)

# Shell 命令安全校验开关（默认开启）
SHELL_SAFETY_STRICT = get_env_bool("SHELL_SAFETY_STRICT", True)
