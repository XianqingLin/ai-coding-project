"""AI Coding TUI 模块.

采用 Textual 构建全屏交互式终端界面.
"""

from typing import Optional

from rich.console import Console

from ai_coding.agent import AgentService
from ai_coding.logger import setup_logging
from ai_coding.tui.app_textual import AICodingApp


def run_tui(
    work_dir: str,
    auto_approve: bool = False,
    session_id: Optional[str] = None,
) -> None:
    """启动 Textual 全屏 TUI."""
    setup_logging()

    service = AgentService(work_dir=work_dir, auto_approve=auto_approve)

    if session_id and not service.switch_session(session_id):
        Console().print(f"[red]Session not found: {session_id}[/red]")
        return

    app = AICodingApp(service)
    app.run()


__all__ = ["run_tui"]
