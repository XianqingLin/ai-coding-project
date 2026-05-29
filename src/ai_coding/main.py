"""程序入口模块.

使用 LangGraph Agent 处理用户请求.
启动时的当前工作目录自动作为项目根路径，Agent 的所有文件操作都在该路径下进行.
"""

import os
import sys
from pathlib import Path

from ai_coding.config import DEFAULT_LLM_PROVIDER
from ai_coding.lc_llm import create_lc_llm
from ai_coding.langgraph_agent import LangGraphAgent
from ai_coding.logger import setup_logging
from ai_coding.tools import DEFAULT_TOOLS


def _resolve_project_root(cli_args: list[str]) -> Path:
    """确定项目根路径.

    规则:
    1. 如果传入路径参数，使用该路径
    2. 如果传入 --interactive，进入交互式选择
    3. 否则使用当前工作目录

    Args:
        cli_args: 命令行参数列表.

    Returns:
        解析后的绝对路径.

    """
    # 交互式模式
    if "--interactive" in cli_args or "-i" in cli_args:
        return _interactive_select_root()

    # 过滤出非选项参数
    positional = [a for a in cli_args if not a.startswith("-")]

    # 显式路径参数
    if positional:
        return _validate_path(Path(positional[0]).resolve())

    # 默认：当前工作目录
    return Path.cwd().resolve()


def _validate_path(path: Path) -> Path:
    """验证路径是否存在且为目录."""
    if not path.exists():
        print(f"[错误] 路径不存在: {path}")
        sys.exit(1)
    if not path.is_dir():
        print(f"[错误] 不是目录: {path}")
        sys.exit(1)
    return path


def _interactive_select_root() -> Path:
    """交互式选择项目根路径."""
    auto_root = _find_git_root(Path.cwd())
    default_hint = str(auto_root) if auto_root else str(Path.cwd())

    print("=" * 50)
    print("  AI Coding - 项目根路径设置")
    print("=" * 50)
    print(f"  默认: {default_hint}")
    print("  提示: 直接回车使用默认值，或输入目标项目路径")
    print("=" * 50)

    try:
        user_input = input("项目根路径: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[信息] 取消启动.")
        sys.exit(0)

    if user_input:
        root = Path(user_input).expanduser().resolve()
    else:
        root = Path(default_hint).resolve()

    return _validate_path(root)


def _find_git_root(start: Path) -> Path | None:
    """向上查找 git 仓库根目录."""
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").is_dir():
            return parent
    return None


def main() -> int:
    """程序主入口.

    Returns:
        退出状态码 (0 表示成功).

    """
    setup_logging()

    # 解析项目根路径
    project_root = _resolve_project_root(sys.argv[1:])

    # 切换到项目目录（工具操作都基于此目录）
    os.chdir(project_root)

    provider = DEFAULT_LLM_PROVIDER

    try:
        llm = create_lc_llm(provider)
    except Exception as e:
        print(f"[Error] {e}")
        return 1

    agent = LangGraphAgent(
        llm=llm,
        tools=DEFAULT_TOOLS,
        max_iterations=10,
        streaming=True,
    )

    from ai_coding.interface.textual_app import ChatApp  # 延迟导入避免循环依赖
    app = ChatApp(
        agent=agent,
        verbose=True,
        startup_info={
            "project": str(project_root),
            "provider": provider,
            "model": llm.model_name,
        },
    )
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
