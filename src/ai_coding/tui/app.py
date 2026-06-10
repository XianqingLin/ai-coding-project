"""AI Coding TUI 主应用 —— prompt_toolkit + Rich 版本.

特点：
- 在终端中原地渲染，不清屏
- Kimi Code CLI 骨架 + Claude Code 风格细节
- 支持 verbose 轨迹输出和流式输出
"""

import os
from typing import Optional

from ai_coding.config import DEFAULT_LLM_PROVIDER
from ai_coding.llm import create_lc_llm
from ai_coding.agent import SessionManager
from ai_coding.logger import setup_logging
from ai_coding.tools import DEFAULT_TOOLS
from ai_coding.tui.shell import run_shell


def run_tui(
    work_dir: str,
    auto_approve: bool = False,
    session_id: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """启动 TUI.

    Args:
        work_dir: 项目工作目录.
        auto_approve: 是否自动批准工具调用.
        session_id: 指定要切换到的会话 ID.
        verbose: 是否启用详细输出.
    """
    setup_logging()
    os.chdir(work_dir)

    sm = SessionManager(
        llm_factory=lambda: create_lc_llm(DEFAULT_LLM_PROVIDER),
        tools=DEFAULT_TOOLS,
        auto_approve=auto_approve,
        work_dir=work_dir,
    )

    if session_id and not sm.switch(session_id):
        from rich.console import Console

        console = Console()
        console.print(f"[red]Session not found: {session_id}[/red]")
        return

    run_shell(sm, verbose=verbose)
