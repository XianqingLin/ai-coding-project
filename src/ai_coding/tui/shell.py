"""prompt_toolkit 输入循环 —— 在终端中原地交互.

特点：
- 不清屏，保留终端历史
- 基于 Rich 的美化输出
- Enter 提交，Shift+Enter 换行（通过多行模式 + 自定义绑定）
- 上下箭头浏览历史
"""

import os
import sys
from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import confirm

from ai_coding.agent import SessionManager
from ai_coding.agent.core import DEFAULT_CONTEXT_LIMIT
from ai_coding.config import DEFAULT_LLM_PROVIDER
from ai_coding.tui.render import (
    console,
    print_help,
    print_message,
    print_sessions,
    print_status_bar,
    print_user_input,
    print_welcome,
)


def _get_history_path() -> str:
    """获取历史记录文件路径."""
    home = Path.home() / ".ai-coding"
    home.mkdir(parents=True, exist_ok=True)
    return str(home / "tui_history")


def _create_key_bindings() -> KeyBindings:
    """创建自定义键绑定.

    Enter: 提交当前输入（默认行为）.
    """
    kb = KeyBindings()

    @kb.add("enter")
    def _(event):
        """Enter 提交当前输入."""
        event.current_buffer.validate_and_handle()

    return kb


def _handle_session_command(sm: SessionManager, parts: list[str]) -> None:
    """处理 /session 子命令."""
    if len(parts) < 2:
        print_message(
            "Usage:\n"
            "  /session list\n"
            "  /session switch <ID>\n"
            "  /session rm <ID>\n"
            "  /session rename <ID> <NAME>",
            role="system",
        )
        return

    sub = parts[1].lower()

    if sub == "list":
        print_sessions(sm.list())
    elif sub == "switch":
        if len(parts) < 3:
            print_message("Usage: /session switch <ID>", role="system")
            return
        sid = parts[2]
        if sm.switch(sid):
            print_message(f"Switched to: {sid}", role="system")
        else:
            print_message(f"Session not found: {sid}", role="error")
    elif sub in ("rm", "delete", "del"):
        if len(parts) < 3:
            print_message("Usage: /session rm <ID>", role="system")
            return
        sid = parts[2]
        if sm.delete(sid):
            print_message(f"Deleted session: {sid}", role="system")
        else:
            print_message(f"Session not found: {sid}", role="error")
    elif sub == "rename":
        if len(parts) < 4:
            print_message("Usage: /session rename <ID> <NAME>", role="system")
            return
        sid = parts[2]
        name = " ".join(parts[3:])
        if sm.rename(sid, name):
            print_message(f"Renamed to: {name}", role="system")
        else:
            print_message(f"Session not found: {sid}", role="error")
    else:
        print_message(f"Unknown subcommand: {sub}", role="error")


def _handle_command(sm: SessionManager, cmd: str, args: str, verbose: bool) -> bool:
    """处理内置命令.

    Returns:
        True 表示继续循环，False 表示退出.
    """
    agent = sm.get_current_agent()

    if cmd in ("/exit", "/quit"):
        print_message("Bye.", role="system")
        return False

    if cmd == "/help":
        print_help()
        return True

    if cmd == "/status":
        if agent is None or not hasattr(agent, "get_context_usage"):
            print_message("Status not available.", role="system")
            return True
        usage = agent.get_context_usage()
        used = usage.get("used_tokens", 0)
        limit = usage.get("limit_tokens", DEFAULT_CONTEXT_LIMIT)
        pct = usage.get("percentage", 0.0)
        bar_len = 20
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print_message(f"Context: [{bar}] {pct}% ({used:,} / {limit:,} tokens)", role="system")
        return True

    if cmd == "/clear":
        console.clear()
        return True

    if cmd == "/new":
        sid = sm.create()
        session = sm.current
        print_message(
            f"New session: {session.name if session else sid}", role="system"
        )
        return True

    if cmd == "/tools":
        if agent is None:
            print_message("No agent available.", role="error")
            return True
        stats = agent.get_stats()
        tools = stats.get("tools", [])
        if not tools:
            print_message("No tools available.", role="system")
            return True
        lines = [f"Available tools ({len(tools)}):"]
        for name in tools:
            lines.append(f"  - {name}")
        print_message("\n".join(lines), role="system")
        return True

    if cmd == "/history":
        if agent is None:
            print_message("No agent available.", role="error")
            return True
        history = agent.get_history()
        if not history:
            print_message("History is empty.", role="system")
            return True
        lines = ["Conversation history:"]
        for i, msg in enumerate(history, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if len(content) > 100:
                content = content[:100] + "..."
            lines.append(f"  {i}. [{role}] {content}")
        print_message("\n".join(lines), role="system")
        return True

    if cmd == "/verbose":
        print_message(
            "Use --verbose flag on startup to control verbose mode.", role="system"
        )
        return True

    if cmd == "/stream":
        if agent is None:
            print_message("No agent available.", role="error")
            return True
        agent.streaming = not agent.streaming
        status = "on" if agent.streaming else "off"
        print_message(f"Streaming mode {status}.", role="system")
        return True

    print_message(f"Unknown command: {cmd}. Type /help for available commands.", role="error")
    return True


def _run_agent_verbose(sm: SessionManager, user_input: str) -> None:
    """以 verbose 模式运行 Agent，打印完整轨迹."""
    agent = sm.get_current_agent()
    if agent is None:
        print_message("No active session.", role="error")
        return

    if not hasattr(agent, "run_with_trace"):
        result = agent.run(user_input)
        print_message(result, role="assistant")
        return

    for event in agent.run_with_trace(user_input):
        etype = event.get("type")
        if etype == "thinking":
            print_message(event.get("text", ""), role="thinking")
        elif etype == "assistant":
            print_message(event.get("text", ""), role="assistant")
        elif etype == "tool_call":
            name = event.get("name", "")
            args = event.get("args", {})
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
            print_message(f"{name}({args_str})", role="tool")
        elif etype == "observation":
            text = event.get("text", "")
            if len(text) > 200:
                text = text[:200] + f" ... ({len(text)} chars)"
            print_message(text, role="system")
        elif etype == "error":
            print_message(event.get("text", ""), role="error")


def _run_agent_stream(sm: SessionManager, user_input: str) -> None:
    """以流式模式运行 Agent."""
    agent = sm.get_current_agent()
    if agent is None:
        print_message("No active session.", role="error")
        return

    # 先打印前缀，然后流式输出
    console.print("[bold green][Assistant][/] ", end="")

    try:
        for chunk in agent.run_stream(user_input):
            console.print(chunk, end="")
        console.print()  # 换行
    except Exception as e:
        console.print()
        print_message(f"Stream error: {e}", role="error")


def _run_agent_normal(sm: SessionManager, user_input: str) -> None:
    """以普通模式运行 Agent."""
    agent = sm.get_current_agent()
    if agent is None:
        print_message("No active session.", role="error")
        return

    try:
        result = agent.run(user_input)
        print_message(result, role="assistant")
    except Exception as e:
        print_message(f"Agent error: {e}", role="error")


def run_shell(sm: SessionManager, verbose: bool = False) -> None:
    """启动 prompt_toolkit 交互式 Shell.

    Args:
        sm: 会话管理器.
        verbose: 是否启用详细输出（显示轨迹）.
    """
    try:
        session = PromptSession(
            message=">>> ",
            multiline=False,
            history=FileHistory(_get_history_path()),
            key_bindings=_create_key_bindings(),
        )
    except Exception as e:
        # 非 TTY 环境回退到普通 REPL
        from ai_coding.main import _run_repl
        _run_repl(sm, verbose=verbose)
        return

    current = sm.current
    session_name = current.name if current else "default"
    print_welcome(sm.work_dir, DEFAULT_LLM_PROVIDER, session_name)

    while True:
        try:
            text = session.prompt()
        except (EOFError, KeyboardInterrupt):
            print_message("Bye.", role="system")
            break

        user_input = text.strip()
        if not user_input:
            continue

        # 打印用户输入
        print_user_input(user_input)

        # 命令处理
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd == "/session":
                _handle_session_command(sm, parts)
                continue

            args = user_input[len(cmd) :].strip()
            if not _handle_command(sm, cmd, args, verbose):
                break
            continue

        # Agent 调用
        agent = sm.get_current_agent()
        if agent is None:
            print_message(
                "No active session. Use /session new to create one.", role="error"
            )
            continue

        if verbose and hasattr(agent, "run_with_trace"):
            _run_agent_verbose(sm, user_input)
        elif agent.streaming:
            _run_agent_stream(sm, user_input)
        else:
            _run_agent_normal(sm, user_input)
