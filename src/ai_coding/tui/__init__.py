"""AI Coding TUI 模块.

采用 Textual 构建全屏交互式终端界面.
"""

from typing import Optional

from rich.console import Console

from ai_coding.agent import SessionManager
from ai_coding.config import DEFAULT_LLM_PROVIDER
from ai_coding.llm import create_lc_llm
from ai_coding.logger import setup_logging
from ai_coding.tools import create_default_tools
from ai_coding.tui.app_textual import AICodingApp


def run_tui(
    work_dir: str,
    auto_approve: bool = False,
    session_id: Optional[str] = None,
) -> None:
    """启动 Textual 全屏 TUI."""
    setup_logging()

    sm = SessionManager(
        llm_factory=lambda: create_lc_llm(DEFAULT_LLM_PROVIDER),
        tools_factory=create_default_tools,
        auto_approve=auto_approve,
        work_dir=work_dir,
    )

    if session_id and not sm.switch(session_id):
        Console().print(f"[red]Session not found: {session_id}[/red]")
        return

    app = AICodingApp(sm)
    app.run()


__all__ = ["run_tui"]
