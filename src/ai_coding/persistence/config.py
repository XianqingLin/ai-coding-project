"""持久化配置.

优先通过环境变量 AI_CODE_HOME 控制存储根目录.
未设置时固定使用项目根目录下的 data/.
"""

import os
from pathlib import Path

_ENV_VAR = "AI_CODE_HOME"


def get_storage_root() -> Path:
    """获取存储根目录.

    优先读取环境变量 AI_CODE_HOME：
    - 相对路径基于当前工作目录解析
    - 绝对路径直接使用
    未设置则固定使用项目根目录下的 data/.
    """
    env = os.getenv(_ENV_VAR, "")
    if env:
        path = Path(os.path.expanduser(env))
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    # 固定到项目根目录下的 data/
    # __file__ 位于 src/ai_coding/persistence/，项目根目录需再向上退一级
    project_root = Path(__file__).parent.parent.parent.parent.resolve()
    return project_root / "data"
