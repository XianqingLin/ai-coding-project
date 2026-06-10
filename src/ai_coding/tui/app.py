"""AI Coding TUI 应用 —— Kimi Code CLI 风格（REPL + bottom_toolbar）.

特点：
- 不清屏，保留终端历史
- 消息像普通终端输出一样向上滚动
- 输入框紧跟在最后一条消息下方
- 底部状态栏显示模型/状态/路径/context
- Welcome 信息作为历史消息的第一条
"""

import os
import threading
from typing import List, Optional, Tuple

from rich.console import Console

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PtkStyle

from ai_coding.agent import SessionManager
from ai_coding.agent.core import DEFAULT_CONTEXT_LIMIT
from ai_coding.config import DEFAULT_LLM_PROVIDER
from ai_coding.llm import create_lc_llm
from ai_coding.logger import setup_logging
from ai_coding.tui.render import render_markdown
from ai_coding.tools import DEFAULT_TOOLS

# Rich console 用于消息输出
console = Console(highlight=False)


def _get_history_path() -> str:
    """获取 prompt_toolkit 历史记录文件路径."""
    from pathlib import Path
    home = Path.home() / ".ai-coding"
    home.mkdir(parents=True, exist_ok=True)
    return str(home / "tui_history")


class AICodingApp:
    """AI Coding TUI 主应用（REPL 风格）."""

    def __init__(
        self,
        sm: SessionManager,
        verbose: bool = False,
    ) -> None:
        self.sm = sm
        self.verbose = verbose
        self._status_text = "Ready"
        self._streaming = False

        self._build_session()
        self._print_welcome()

    # ------------------------------------------------------------------ #
    # 会话构建
    # ------------------------------------------------------------------ #

    def _build_session(self) -> None:
        """构建 PromptSession."""
        style = PtkStyle.from_dict({
            "prompt": "#f9e2af bold",
            "bottom-toolbar": "bg:#11111b #6c7086",
            "bottom-toolbar.text": "#6c7086",
        })

        self.session = PromptSession(
            message=HTML("<prompt>></prompt> "),
            multiline=False,
            history=FileHistory(_get_history_path()),
            bottom_toolbar=self._get_bottom_toolbar,
            style=style,
        )

    def _get_bottom_toolbar(self) -> HTML:
        """生成底部状态栏文本."""
        agent = self.sm.get_current_agent()
        context_info = ""
        if agent and hasattr(agent, "get_context_usage"):
            try:
                usage = agent.get_context_usage()
                pct = usage.get("percentage", 0.0)
                used = usage.get("used_tokens", 0)
                limit = usage.get("limit_tokens", 0)
                context_info = (
                    f"  context: {pct}% ({used / 1000:.1f}k/{limit / 1000:.1f}k)"
                )
            except Exception:
                pass

        left = f"{DEFAULT_LLM_PROVIDER}  {self._status_text}  {self.sm.work_dir}"
        right = context_info
        return HTML(f"<bottom-toolbar>{left}{right}</bottom-toolbar>")

    # ------------------------------------------------------------------ #
    # 渲染输出
    # ------------------------------------------------------------------ #

    def _print_welcome(self) -> None:
        """打印欢迎信息（作为消息历史的第一条）."""
        current = self.sm.current
        session_name = current.name if current else "default"
        lines = [
            "[bold #89b4fa]Welcome to AI Coding![/bold #89b4fa]",
            "[dim]Send /help for help information.[/dim]",
            "",
            f"[dim]Directory:[/dim] {self.sm.work_dir}",
            f"[dim]Session:[/dim]   {session_name}",
            f"[dim]Model:[/dim]     {DEFAULT_LLM_PROVIDER}",
        ]
        # 用蓝色边框的 Panel 样式模拟 Kimi Code CLI 的 Welcome Panel
        from rich.panel import Panel
        welcome_text = "\n".join(lines)
        console.print(Panel(welcome_text, border_style="#4a90e2", padding=(0, 1)))
        console.print()

    def _print_user_input(self, text: str) -> None:
        """打印用户输入（黄色 > 前缀）."""
        console.print(f"[bold #f9e2af]> {text}[/bold #f9e2af]")

    def _print_assistant(self, text: str) -> None:
        """打印 AI 回复（白色 ● 前缀 + Rich 渲染）."""
        renderable = render_markdown(text)
        # 先打印前缀，然后渲染内容
        console.print("[#cdd6f4]● [/#cdd6f4]", end="")
        console.print(renderable)

    def _print_thinking(self, text: str) -> None:
        """打印思考过程（灰色斜体 ● 前缀）."""
        console.print(f"[italic #6c7086]● {text}[/italic #6c7086]")

    def _print_tool(self, text: str) -> None:
        """打印工具调用（黄色 ● 前缀）."""
        console.print(f"[#f9e2af]● {text}[/#f9e2af]")

    def _print_error(self, text: str) -> None:
        """打印错误（红色 ● 前缀）."""
        console.print(f"[#f38ba8]● {text}[/#f38ba8]")

    def _print_system(self, text: str) -> None:
        """打印系统消息（灰色 ● 前缀）."""
        console.print(f"[#6c7086]● {text}[/#6c7086]")

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """启动主循环."""
        while True:
            try:
                text = self.session.prompt()
            except (EOFError, KeyboardInterrupt):
                console.print()
                console.print("[dim]Bye.[/dim]")
                break

            user_input = text.strip()
            if not user_input:
                continue

            # 打印用户输入
            self._print_user_input(user_input)

            # 处理命令
            if user_input.startswith("/"):
                self._handle_command(user_input)
                continue

            # 运行 Agent
            self._run_agent(user_input)

    # ------------------------------------------------------------------ #
    # 命令处理
    # ------------------------------------------------------------------ #

    def _handle_command(self, text: str) -> None:
        """处理内置命令."""
        parts = text.split()
        cmd = parts[0].lower()

        if cmd in ("/exit", "/quit"):
            console.print("[dim]Bye.[/dim]")
            raise SystemExit(0)

        if cmd == "/help":
            self._print_system(self._help_text())
            return

        if cmd == "/new":
            sid = self.sm.create()
            self._print_system(f"New session: {sid}")
            return

        if cmd == "/status":
            agent = self.sm.get_current_agent()
            if agent and hasattr(agent, "get_context_usage"):
                usage = agent.get_context_usage()
                used = usage.get("used_tokens", 0)
                limit = usage.get("limit_tokens", DEFAULT_CONTEXT_LIMIT)
                pct = usage.get("percentage", 0.0)
                bar_len = 20
                filled = int(bar_len * pct / 100)
                bar = "█" * filled + "░" * (bar_len - filled)
                self._print_system(
                    f"Context: [{bar}] {pct}% ({used:,} / {limit:,} tokens)"
                )
            else:
                self._print_system("Status not available.")
            return

        if cmd == "/clear":
            console.clear()
            self._print_welcome()
            return

        if cmd == "/session":
            self._handle_session_command(parts)
            return

        if cmd == "/tools":
            agent = self.sm.get_current_agent()
            if agent:
                stats = agent.get_stats()
                tools = stats.get("tools", [])
                lines = [f"Available tools ({len(tools)}):"]
                for name in tools:
                    lines.append(f"  - {name}")
                self._print_system("\n".join(lines))
            else:
                self._print_error("No agent available.")
            return

        if cmd == "/history":
            agent = self.sm.get_current_agent()
            if agent:
                history = agent.get_history()
                if not history:
                    self._print_system("History is empty.")
                    return
                lines = ["Conversation history:"]
                for i, msg in enumerate(history, 1):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if len(content) > 100:
                        content = content[:100] + "..."
                    lines.append(f"  {i}. [{role}] {content}")
                self._print_system("\n".join(lines))
            else:
                self._print_error("No agent available.")
            return

        if cmd == "/stream":
            agent = self.sm.get_current_agent()
            if agent:
                agent.streaming = not agent.streaming
                status = "on" if agent.streaming else "off"
                self._print_system(f"Streaming mode {status}.")
            return

        self._print_error(
            f"Unknown command: {cmd}. Type /help for available commands."
        )

    def _handle_session_command(self, parts: List[str]) -> None:
        """处理 /session 子命令."""
        if len(parts) < 2:
            self._print_system(
                "Usage:\n"
                "  /session list\n"
                "  /session switch <ID>\n"
                "  /session rm <ID>\n"
                "  /session rename <ID> <NAME>"
            )
            return

        sub = parts[1].lower()
        if sub == "list":
            sessions = self.sm.list()
            if not sessions:
                self._print_system("No sessions.")
                return
            lines = [f"Sessions ({len(sessions)}):"]
            for s in sessions:
                marker = "*" if s.get("is_current") else " "
                lines.append(
                    f"  [{marker}] {s['session_id']}  {s['name']}  "
                    f"({s.get('message_count', 0)} msgs)"
                )
            self._print_system("\n".join(lines))
        elif sub == "switch":
            if len(parts) < 3:
                self._print_system("Usage: /session switch <ID>")
                return
            sid = parts[2]
            if self.sm.switch(sid):
                self._print_system(f"Switched to: {sid}")
            else:
                self._print_error(f"Session not found: {sid}")
        elif sub in ("rm", "delete", "del"):
            if len(parts) < 3:
                self._print_system("Usage: /session rm <ID>")
                return
            sid = parts[2]
            if self.sm.delete(sid):
                self._print_system(f"Deleted session: {sid}")
            else:
                self._print_error(f"Session not found: {sid}")
        elif sub == "rename":
            if len(parts) < 4:
                self._print_system("Usage: /session rename <ID> <NAME>")
                return
            sid = parts[2]
            name = " ".join(parts[3:])
            if self.sm.rename(sid, name):
                self._print_system(f"Renamed to: {name}")
            else:
                self._print_error(f"Session not found: {sid}")
        else:
            self._print_error(f"Unknown subcommand: {sub}")

    # ------------------------------------------------------------------ #
    # Agent 调用
    # ------------------------------------------------------------------ #

    def _run_agent(self, text: str) -> None:
        """运行 Agent（阻塞当前线程）."""
        self._status_text = "Thinking..."
        self.session.app.invalidate()  # 刷新 bottom_toolbar

        agent = self.sm.get_current_agent()
        if agent is None:
            self._print_error("No active session.")
            self._status_text = "Ready"
            return

        try:
            if self.verbose and hasattr(agent, "run_with_trace"):
                self._run_agent_verbose(text)
            elif agent.streaming:
                self._run_agent_stream(text)
            else:
                result = agent.run(text)
                self._print_assistant(result)
        except Exception as e:
            self._print_error(f"Agent error: {e}")
        finally:
            self._status_text = "Ready"
            self.session.app.invalidate()  # 刷新 bottom_toolbar

    def _run_agent_verbose(self, text: str) -> None:
        """以 verbose 模式运行 Agent."""
        agent = self.sm.get_current_agent()
        for event in agent.run_with_trace(text):
            etype = event.get("type")
            if etype == "thinking":
                self._print_thinking(event.get("text", ""))
            elif etype == "assistant":
                self._print_assistant(event.get("text", ""))
            elif etype == "tool_call":
                name = event.get("name", "")
                args = event.get("args", {})
                args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                self._print_tool(f"{name}({args_str})")
            elif etype == "observation":
                t = event.get("text", "")
                if len(t) > 200:
                    t = t[:200] + f" ... ({len(t)} chars)"
                self._print_system(t)
            elif etype == "error":
                self._print_error(event.get("text", ""))

    def _run_agent_stream(self, text: str) -> None:
        """以流式模式运行 Agent."""
        agent = self.sm.get_current_agent()
        # 打印前缀
        console.print("[#cdd6f4]● [/#cdd6f4]", end="")
        try:
            for chunk in agent.run_stream(text):
                console.print(chunk, end="")
            console.print()  # 最终换行
        except Exception as e:
            console.print()
            self._print_error(f"Stream error: {e}")

    @staticmethod
    def _help_text() -> str:
        """返回帮助文本."""
        return (
            "Available commands:\n"
            "  /help              — Show this help\n"
            "  /new               — Create a new session\n"
            "  /session list      — List all sessions\n"
            "  /session switch ID — Switch session\n"
            "  /session rm ID     — Remove session\n"
            "  /session rename ID NAME\n"
            "  /status            — Show context usage\n"
            "  /tools             — List available tools\n"
            "  /history           — Show conversation history\n"
            "  /clear             — Clear screen\n"
            "  /stream            — Toggle streaming mode\n"
            "  /exit              — Exit\n"
            "  (Any other text is sent to the Agent)"
        )


def run_tui(
    work_dir: str,
    auto_approve: bool = False,
    session_id: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """启动 TUI."""
    setup_logging()
    os.chdir(work_dir)

    sm = SessionManager(
        llm_factory=lambda: create_lc_llm(DEFAULT_LLM_PROVIDER),
        tools=DEFAULT_TOOLS,
        auto_approve=auto_approve,
        work_dir=work_dir,
    )

    if session_id and not sm.switch(session_id):
        console.print(f"[red]Session not found: {session_id}[/red]")
        return

    app = AICodingApp(sm, verbose=verbose)
    app.run()
