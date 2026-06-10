"""AI Coding CLI 入口模块.

使用 Typer 提供命令行接口，支持一次性问答、交互式聊天和会话管理.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import typer

from ai_coding.config import DEFAULT_LLM_PROVIDER
from ai_coding.llm import create_lc_llm
from ai_coding.agent import SessionManager
from ai_coding.logger import setup_logging, get_logger
from ai_coding.tools import DEFAULT_TOOLS
from ai_coding.main import _print_welcome, _run_repl

logger = get_logger(__name__)

app = typer.Typer(help="AI Coding Assistant CLI")
session_app = typer.Typer(help="Session management commands")
app.add_typer(session_app, name="session")


def _resolve_work_dir(work_dir: str) -> str:
    """解析并验证工作目录."""
    path = Path(work_dir).expanduser().resolve()
    if not path.exists():
        typer.echo(f"[错误] 路径不存在: {path}", err=True)
        raise typer.Exit(1)
    if not path.is_dir():
        typer.echo(f"[错误] 不是目录: {path}", err=True)
        raise typer.Exit(1)
    return str(path)


def _create_sm(work_dir: str, auto_approve: bool = False) -> SessionManager:
    """创建 SessionManager 实例."""
    provider = DEFAULT_LLM_PROVIDER
    os.chdir(work_dir)
    return SessionManager(
        llm_factory=lambda: create_lc_llm(provider),
        tools=DEFAULT_TOOLS,
        auto_approve=auto_approve,
        work_dir=work_dir,
    )


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="要发送给 Agent 的指令"),
    work_dir: str = typer.Option(".", "--work-dir", "-w", help="项目工作目录"),
    auto_approve: bool = typer.Option(False, "--auto-approve", "-a", help="自动批准工具调用"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="指定会话 ID"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="显示详细输出"),
):
    """向 Agent 发送一次性指令并打印回复."""
    setup_logging()
    work_dir = _resolve_work_dir(work_dir)
    sm = _create_sm(work_dir, auto_approve)

    if session:
        if not sm.switch(session):
            typer.echo(f"[错误] 会话不存在: {session}", err=True)
            raise typer.Exit(1)

    agent = sm.get_current_agent()
    if agent is None:
        typer.echo("[错误] 当前没有活跃的会话", err=True)
        raise typer.Exit(1)

    if verbose:
        typer.echo(f"Work dir: {work_dir}")
        typer.echo(f"Session: {sm.current.name if sm.current else 'none'}")
        typer.echo("---")

    result = agent.run(prompt)
    typer.echo(result)


@app.command()
def chat(
    work_dir: str = typer.Option(".", "--work-dir", "-w", help="项目工作目录"),
    auto_approve: bool = typer.Option(False, "--auto-approve", "-a", help="自动批准工具调用"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="显示详细输出"),
    no_tui: bool = typer.Option(False, "--no-tui", help="使用旧版文本 REPL"),
):
    """启动交互式聊天会话."""
    setup_logging()
    work_dir = _resolve_work_dir(work_dir)

    if no_tui:
        from ai_coding.main import start_chat_session
        code = start_chat_session(work_dir, auto_approve=auto_approve, verbose=verbose)
        raise typer.Exit(code)

    from ai_coding.tui import run_tui
    run_tui(work_dir=work_dir, auto_approve=auto_approve, verbose=verbose)


@session_app.command("list")
def session_list(
    work_dir: str = typer.Option(".", "--work-dir", "-w", help="项目工作目录"),
):
    """列出所有会话."""
    work_dir = _resolve_work_dir(work_dir)
    sm = _create_sm(work_dir)
    sessions = sm.list()
    if not sessions:
        typer.echo("No sessions.")
        return
    for s in sessions:
        marker = "*" if s["is_current"] else " "
        typer.echo(
            f" [{marker}] {s['session_id']}  {s['name']}  ({s['message_count']} msgs)"
        )


@session_app.command("new")
def session_new(
    name: str = typer.Argument("default", help="会话名称"),
    work_dir: str = typer.Option(".", "--work-dir", "-w", help="项目工作目录"),
):
    """创建新会话."""
    work_dir = _resolve_work_dir(work_dir)
    sm = _create_sm(work_dir)
    sid = sm.create(name=name)
    typer.echo(f"Created session: {sid}")


@session_app.command("switch")
def session_switch(
    session_id: str = typer.Argument(..., help="要切换到的会话 ID"),
    work_dir: str = typer.Option(".", "--work-dir", "-w", help="项目工作目录"),
):
    """切换到指定会话."""
    work_dir = _resolve_work_dir(work_dir)
    sm = _create_sm(work_dir)
    if sm.switch(session_id):
        typer.echo(f"Switched to: {session_id}")
    else:
        typer.echo(f"Session not found: {session_id}", err=True)
        raise typer.Exit(1)


@session_app.command("delete")
def session_delete(
    session_id: str = typer.Argument(..., help="要删除的会话 ID"),
    work_dir: str = typer.Option(".", "--work-dir", "-w", help="项目工作目录"),
):
    """删除指定会话."""
    work_dir = _resolve_work_dir(work_dir)
    sm = _create_sm(work_dir)
    if sm.delete(session_id):
        typer.echo(f"Deleted session: {session_id}")
    else:
        typer.echo(f"Session not found: {session_id}", err=True)
        raise typer.Exit(1)


def main() -> None:
    """CLI 入口."""
    app()


if __name__ == "__main__":
    main()
