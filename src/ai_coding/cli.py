"""AI Coding CLI 入口模块.

使用 Typer 提供命令行接口，支持一次性问答、交互式聊天和会话管理.
"""

from pathlib import Path
from typing import Optional

import typer

from ai_coding.agent import AgentService
from ai_coding.logger import setup_logging, get_logger


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


def _create_service(work_dir: str, auto_approve: bool = False) -> AgentService:
    """创建 AgentService 实例."""
    return AgentService(work_dir=work_dir, auto_approve=auto_approve)


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="要发送给 Agent 的指令"),
    work_dir: str = typer.Option(".", "--work-dir", "-w", help="项目工作目录"),
    auto_approve: bool = typer.Option(False, "--auto-approve", "-a", help="自动批准工具调用"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="指定会话 ID"),
):
    """向 Agent 发送一次性指令并打印回复."""
    setup_logging()
    work_dir = _resolve_work_dir(work_dir)
    service = _create_service(work_dir, auto_approve)

    if session:
        if not service.switch_session(session):
            typer.echo(f"[错误] 会话不存在: {session}", err=True)
            raise typer.Exit(1)

    result = service.send_message(prompt)
    typer.echo(result)


@app.command()
def chat(
    work_dir: str = typer.Option(".", "--work-dir", "-w", help="项目工作目录"),
    auto_approve: bool = typer.Option(False, "--auto-approve", "-a", help="自动批准工具调用"),
):
    """启动交互式聊天会话（固定为流式+verbose 模式）."""
    setup_logging()
    work_dir = _resolve_work_dir(work_dir)

    from ai_coding.tui import run_tui
    run_tui(work_dir=work_dir, auto_approve=auto_approve)


@session_app.command("list")
def session_list(
    work_dir: str = typer.Option(".", "--work-dir", "-w", help="项目工作目录"),
):
    """列出所有会话."""
    work_dir = _resolve_work_dir(work_dir)
    service = _create_service(work_dir)
    sessions = service.list_sessions()
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
    service = _create_service(work_dir)
    sid = service.create_session(name=name)
    typer.echo(f"Created session: {sid}")


@session_app.command("switch")
def session_switch(
    session_id: str = typer.Argument(..., help="要切换到的会话 ID"),
    work_dir: str = typer.Option(".", "--work-dir", "-w", help="项目工作目录"),
):
    """切换到指定会话."""
    work_dir = _resolve_work_dir(work_dir)
    service = _create_service(work_dir)
    if service.switch_session(session_id):
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
    service = _create_service(work_dir)
    if service.delete_session(session_id):
        typer.echo(f"Deleted session: {session_id}")
    else:
        typer.echo(f"Session not found: {session_id}", err=True)
        raise typer.Exit(1)


def _configure_windows_encoding() -> None:
    """在 Windows 上强制 stdout/stderr 使用 UTF-8 编码.

    Windows 终端默认使用 GBK（代码页 936），当输出包含 emoji 或其他非
    ASCII 字符时，typer/click 的 echo 会抛出 UnicodeEncodeError。
    在 CLI 入口尽早将标准流重配置为 UTF-8，避免乱码和崩溃。
    """
    import sys

    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                # 某些测试或重定向场景下 reconfigure 可能不可用，忽略
                pass


def main() -> None:
    """CLI 入口.

    无参数时默认执行 chat 命令.
    """
    import sys

    _configure_windows_encoding()

    if len(sys.argv) == 1:
        sys.argv.append("chat")
    elif len(sys.argv) > 1 and sys.argv[1].startswith("-") and sys.argv[1] not in ("--help", "-h", "--version"):
        sys.argv.insert(1, "chat")

    app()


if __name__ == "__main__":
    main()
