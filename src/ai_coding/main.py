"""程序入口模块.

使用 LangGraph Agent 处理用户请求.
启动时的当前工作目录自动作为项目根路径，Agent 的所有文件操作都在该路径下进行.
"""

import os
import sys
from pathlib import Path

from ai_coding.config import DEFAULT_LLM_PROVIDER
from ai_coding.llm import create_lc_llm
from ai_coding.agent import LangGraphAgent, SessionManager
from ai_coding.logger import setup_logging, get_logger
from ai_coding.tools import DEFAULT_TOOLS

logger = get_logger(__name__)


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


def _print_welcome(
    project_root: Path, provider: str, model: str, session_name: str
) -> None:
    """打印欢迎信息."""
    print("=" * 50)
    print("  AI Coding Assistant")
    print("=" * 50)
    print(f"  Project : {project_root}")
    print(f"  Provider: {provider}")
    print(f"  Model   : {model}")
    print(f"  Session : {session_name}")
    print("-" * 50)
    print("  Commands:")
    print("    /help              - Show help")
    print("    /status            - Show context usage")
    print("    /tools             - List available tools")
    print("    /history           - Show conversation history")
    print("    /new               - Create a new session")
    print("    /session list      - List all sessions")
    print("    /session new NAME  - Create named session")
    print("    /session switch ID - Switch to session")
    print("    /session rm ID     - Remove session")
    print("    /session rename ID NAME")
    print("    /verbose           - Toggle verbose mode")
    print("    /stream            - Toggle streaming mode")
    print("    /clear             - Clear screen")
    print("    /exit              - Exit")
    print("=" * 50)


def _print_event(event: dict) -> None:
    """打印 Agent 轨迹事件."""
    etype = event.get("type")
    if etype == "thinking":
        print(f"\n[Thinking] {event.get('text', '')}")
    elif etype == "assistant":
        print(f"\n[Assistant] {event.get('text', '')}")
    elif etype == "tool_call":
        name = event.get("name", "")
        args = event.get("args", {})
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        print(f"\n[Tool] {name}({args_str})")
    elif etype == "observation":
        text = event.get("text", "")
        if len(text) > 200:
            text = text[:200] + f" ... ({len(text)} chars)"
        print(f"\n[Observation] {text}")
    elif etype == "error":
        print(f"\n[Error] {event.get('text', '')}")


def _handle_session_command(sm: SessionManager, cmd_parts: list[str]) -> str:
    """处理 /session 子命令."""
    if len(cmd_parts) < 2:
        return (
            "Usage:\n"
            "  /session list\n"
            "  /session new [NAME]\n"
            "  /session switch <ID>\n"
            "  /session rm <ID>\n"
            "  /session rename <ID> <NAME>"
        )

    sub = cmd_parts[1].lower()

    if sub == "list":
        sessions = sm.list()
        if not sessions:
            return "No sessions."
        lines = [f"Sessions ({len(sessions)}):"]
        for s in sessions:
            marker = "*" if s["is_current"] else " "
            lines.append(
                f"  [{marker}] {s['session_id']}  {s['name']}  "
                f"({s['message_count']} msgs)"
            )
        return "\n".join(lines)

    if sub == "new":
        name = " ".join(cmd_parts[2:]) if len(cmd_parts) > 2 else ""
        sid = sm.create(name=name)
        return f"Created session: {sid}"

    if sub == "switch":
        if len(cmd_parts) < 3:
            return "Usage: /session switch <ID>"
        sid = cmd_parts[2]
        if sm.switch(sid):
            session = sm.current
            return f"Switched to: {session.name if session else sid}"
        return f"Session not found: {sid}"

    if sub in ("rm", "delete", "del"):
        if len(cmd_parts) < 3:
            return "Usage: /session rm <ID>"
        sid = cmd_parts[2]
        if sm.delete(sid):
            return f"Deleted session: {sid}"
        return f"Session not found: {sid}"

    if sub == "rename":
        if len(cmd_parts) < 4:
            return "Usage: /session rename <ID> <NAME>"
        sid = cmd_parts[2]
        name = " ".join(cmd_parts[3:])
        if sm.rename(sid, name):
            return f"Renamed to: {name}"
        return f"Session not found: {sid}"

    return f"Unknown /session subcommand: {sub}"


def _handle_command(
    sm: SessionManager, cmd: str, args: str
) -> str:
    """处理内置命令.

    Returns:
        命令执行结果文本，空字符串表示无需输出.

    """
    agent = sm.get_current_agent()
    if agent is None:
        return "[错误] 当前没有活跃的会话."

    if cmd in ("/exit", "/quit"):
        print("\n[信息] 退出.")
        sys.exit(0)

    if cmd == "/help":
        return (
            "Available commands:\n"
            "  /help              - Show help\n"
            "  /status            - Show context usage\n"
            "  /tools             - List available tools\n"
            "  /history           - Show conversation history\n"
            "  /new               - Create a new session\n"
            "  /session list      - List all sessions\n"
            "  /session new NAME  - Create named session\n"
            "  /session switch ID - Switch to session\n"
            "  /session rm ID     - Remove session\n"
            "  /session rename ID NAME\n"
            "  /verbose           - Toggle verbose mode\n"
            "  /stream            - Toggle streaming mode\n"
            "  /clear             - Clear screen\n"
            "  /exit              - Exit"
        )

    if cmd == "/status":
        if not hasattr(agent, "get_context_usage"):
            return "Status not available."
        usage = agent.get_context_usage()
        used = usage.get("used_tokens", 0)
        limit = usage.get("limit_tokens", 128000)
        pct = usage.get("percentage", 0.0)
        bar_len = 20
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        return f"Context: [{bar}] {pct}% ({used:,} / {limit:,} tokens)"

    if cmd == "/clear":
        print("\n" * 50)
        return "Screen cleared."

    if cmd == "/new":
        sid = sm.create()
        session = sm.current
        return f"New session: {session.name if session else sid}"

    if cmd == "/tools":
        stats = agent.get_stats()
        tools = stats.get("tools", [])
        if not tools:
            return "No tools available."
        lines = [f"Available tools ({len(tools)}):"]
        for name in tools:
            lines.append(f"  - {name}")
        return "\n".join(lines)

    if cmd == "/history":
        history = agent.get_history()
        if not history:
            return "History is empty."
        lines = ["Conversation history:"]
        for i, msg in enumerate(history, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if len(content) > 100:
                content = content[:100] + "..."
            lines.append(f"  {i}. [{role}] {content}")
        return "\n".join(lines)

    if cmd == "/verbose":
        return "Use --verbose flag on startup to control verbose mode."

    if cmd == "/stream":
        agent.streaming = not agent.streaming
        status = "on" if agent.streaming else "off"
        return f"Streaming mode {status}."

    return f"Unknown command: {cmd}. Type /help for available commands."


def _run_repl(sm: SessionManager, verbose: bool = False) -> None:
    """运行纯文本 REPL 交互循环."""
    while True:
        try:
            user_input = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[信息] 退出.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd == "/session":
                result = _handle_session_command(sm, parts)
                if result:
                    print(f"\n{result}")
                continue

            args = user_input[len(cmd):].strip()
            result = _handle_command(sm, cmd, args)
            if result:
                print(f"\n{result}")
            continue

        agent = sm.get_current_agent()
        if agent is None:
            print("\n[错误] 当前没有活跃的会话，使用 /session new 创建.")
            continue

        logger.info(f"User input: {user_input}")

        if verbose and hasattr(agent, "run_with_trace"):
            for event in agent.run_with_trace(user_input):
                _print_event(event)
        elif agent.streaming:
            print("\n[Assistant] ", end="", flush=True)
            for chunk in agent.run_stream(user_input):
                print(chunk, end="", flush=True)
            print()
        else:
            result = agent.run(user_input)
            print(f"\n[Assistant] {result}")


def main() -> int:
    """程序主入口.

    Returns:
        退出状态码 (0 表示成功).

    """
    setup_logging()

    # 解析命令行参数
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    auto_approve = "--auto-approve" in sys.argv
    cli_args = [a for a in sys.argv[1:] if a not in ("--verbose", "-v", "--auto-approve")]

    # 解析项目根路径
    project_root = _resolve_project_root(cli_args)

    # 切换到项目目录（工具操作都基于此目录）
    os.chdir(project_root)

    provider = DEFAULT_LLM_PROVIDER

    try:
        llm = create_lc_llm(provider)
    except Exception as e:
        print(f"[Error] {e}")
        return 1

    # 使用 SessionManager 管理多会话
    sm = SessionManager(
        llm_factory=lambda: create_lc_llm(provider),
        tools=DEFAULT_TOOLS,
        auto_approve=auto_approve,
    )

    current = sm.current
    session_name = current.name if current else "default"

    _print_welcome(project_root, provider, llm.model_name, session_name)
    _run_repl(sm, verbose=verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
