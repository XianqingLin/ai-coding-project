"""AI Coding TUI 应用 —— Kimi Code CLI 风格.

采用 prompt_toolkit Application 构建固定布局：
- 顶部欢迎卡片（带边框）
- 中间历史消息区（可滚动）
- 底部输入框（带边框和 > 前缀）
- 最底部状态栏
"""

import os
import threading
from typing import Any, Callable, List, Optional, Tuple

from rich.console import Console

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.styles import Style as PtkStyle
from prompt_toolkit.widgets import Frame

from ai_coding.agent import SessionManager
from ai_coding.agent.core import DEFAULT_CONTEXT_LIMIT
from ai_coding.config import DEFAULT_LLM_PROVIDER
from ai_coding.llm import create_lc_llm
from ai_coding.logger import setup_logging
from ai_coding.tui.render import render_markdown
from ai_coding.tools import DEFAULT_TOOLS


class AICodingApp:
    """AI Coding TUI 主应用."""

    def __init__(
        self,
        sm: SessionManager,
        verbose: bool = False,
    ) -> None:
        self.sm = sm
        self.verbose = verbose
        self._status_text = "Ready"
        self._history_entries: List[Tuple[str, str]] = []
        self._console_width = 80
        self._history_console = self._make_console(80)

        self._build_layout()

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    def _make_console(self, width: int) -> Console:
        """创建用于历史渲染的 Rich Console."""
        return Console(
            force_terminal=True,
            color_system="standard",
            width=width,
            highlight=False,
        )

    def _update_console_width(self) -> None:
        """根据终端宽度更新 Console."""
        try:
            width = self.app.output.get_size().columns
            width = max(width - 4, 40)
        except Exception:
            width = 80
        if width != self._console_width:
            self._console_width = width
            self._history_console = self._make_console(width)

    # ------------------------------------------------------------------ #
    # 布局构建
    # ------------------------------------------------------------------ #

    def _build_layout(self) -> None:
        """构建 prompt_toolkit Application 布局."""

        # 欢迎卡片
        self.welcome_control = FormattedTextControl(self._get_welcome_text)
        welcome_window = Window(
            content=self.welcome_control,
            height=6,
            style="class:welcome",
        )
        welcome_frame = Frame(
            body=welcome_window,
            style="class:welcome",
        )

        # 历史消息区
        self.history_control = FormattedTextControl(self._get_history_text)
        self.history_window = Window(
            content=self.history_control,
            wrap_lines=True,
            always_hide_cursor=True,
            style="class:history",
        )

        # 输入框
        self.input_buffer = Buffer(multiline=True)
        self.input_buffer.accept_handler = self._on_accept
        input_control = BufferControl(
            buffer=self.input_buffer,
            input_processors=[BeforeInput("> ")],
            key_bindings=self._create_input_kb(),
        )
        input_window = Window(
            content=input_control,
            height=3,
            wrap_lines=True,
            style="class:input",
        )
        input_frame = Frame(
            body=input_window,
            style="class:input",
        )

        # 状态栏
        self.status_control = FormattedTextControl(self._get_status_text)
        status_window = Window(
            content=self.status_control,
            height=1,
            style="class:statusbar",
        )

        # 全局键绑定
        kb = KeyBindings()

        @kb.add("c-c")
        @kb.add("c-q")
        def _(event):
            """Ctrl+C / Ctrl+Q 退出."""
            event.app.exit()

        # 布局
        layout = Layout(
            HSplit([
                welcome_frame,
                self.history_window,
                input_frame,
                status_window,
            ])
        )

        self.app = Application(
            layout=layout,
            key_bindings=kb,
            full_screen=True,
            style=self._get_style(),
            mouse_support=True,
        )

    def _create_input_kb(self) -> KeyBindings:
        """创建输入框键绑定."""
        kb = KeyBindings()

        @kb.add("enter")
        def _(event):
            """Enter 提交."""
            event.current_buffer.validate_and_handle()

        @kb.add("c-j")
        def _(event):
            """Ctrl+J 插入换行."""
            event.current_buffer.insert_text("\n")

        return kb

    def _get_style(self) -> PtkStyle:
        """定义 prompt_toolkit 样式."""
        return PtkStyle.from_dict({
            "welcome": "bg:#1e1e2e fg:#cdd6f4",
            "welcome-title": "bold fg:#89b4fa",
            "welcome-label": "fg:#6c7086",
            "welcome-value": "fg:#cdd6f4",
            "history": "bg:#181825 fg:#cdd6f4",
            "user": "bold fg:#f9e2af",
            "assistant": "fg:#cdd6f4",
            "thinking": "italic fg:#6c7086",
            "tool": "fg:#f9e2af",
            "error": "fg:#f38ba8",
            "system": "fg:#6c7086",
            "input": "bg:#1e1e2e fg:#cdd6f4",
            "statusbar": "bg:#11111b fg:#6c7086",
            "frame.border": "#45475a",
        })

    # ------------------------------------------------------------------ #
    # 内容生成
    # ------------------------------------------------------------------ #

    def _get_welcome_text(self) -> List[Tuple[str, str]]:
        """生成欢迎文本."""
        current = self.sm.current
        session_name = current.name if current else "default"
        return [
            ("class:welcome-title", "Welcome to AI Coding!\n"),
            ("", "Send /help for help information.\n\n"),
            ("class:welcome-label", "Directory: "),
            ("class:welcome-value", f"{self.sm.work_dir}\n"),
            ("class:welcome-label", "Session:   "),
            ("class:welcome-value", f"{session_name}\n"),
            ("class:welcome-label", "Model:     "),
            ("class:welcome-value", f"{DEFAULT_LLM_PROVIDER}\n"),
        ]

    def _get_history_text(self) -> ANSI:
        """生成历史消息文本（Rich ANSI）."""
        if not self._history_entries:
            return ANSI("")

        self._update_console_width()

        with self._history_console.capture() as capture:
            for role, text in self._history_entries:
                self._render_entry_to_console(role, text)

        return ANSI(capture.get())

    def _render_entry_to_console(self, role: str, text: str) -> None:
        """将单条消息渲染到历史 Console."""
        if role == "user":
            self._history_console.print(f"[bold #f9e2af]> {text}[/]")
        elif role == "assistant":
            renderable = render_markdown(text)
            self._history_console.print(renderable)
        elif role == "thinking":
            self._history_console.print(f"[italic #6c7086]● {text}[/]")
        elif role == "tool":
            self._history_console.print(f"[#f9e2af]● {text}[/]")
        elif role == "error":
            self._history_console.print(f"[#f38ba8]● {text}[/]")
        elif role == "system":
            self._history_console.print(f"[#6c7086]● {text}[/]")
        else:
            self._history_console.print(text)

    def _get_status_text(self) -> List[Tuple[str, str]]:
        """生成状态栏文本（左右对齐）."""
        current = self.sm.current
        session_name = current.name if current else "default"
        agent = self.sm.get_current_agent()

        # Context 使用率
        context_info = ""
        if agent and hasattr(agent, "get_context_usage"):
            try:
                usage = agent.get_context_usage()
                pct = usage.get("percentage", 0.0)
                used = usage.get("used_tokens", 0)
                limit = usage.get("limit_tokens", 0)
                context_info = (
                    f"context: {pct}% ({used / 1000:.1f}k/{limit / 1000:.1f}k)"
                )
            except Exception:
                pass

        left = f"{DEFAULT_LLM_PROVIDER}  {self._status_text}  {self.sm.work_dir}"
        right = context_info

        # 计算填充空格以实现右对齐
        try:
            width = self.app.output.get_size().columns
        except Exception:
            width = 80
        padding_len = max(width - len(left) - len(right), 1)

        return [
            ("class:statusbar", left),
            ("", " " * padding_len),
            ("class:statusbar", right),
        ]

    # ------------------------------------------------------------------ #
    # 交互逻辑
    # ------------------------------------------------------------------ #

    def _on_accept(self, buffer: Buffer) -> bool:
        """处理输入提交."""
        text = buffer.text.strip()
        if not text:
            return True

        buffer.text = ""

        # 显示用户输入
        self._add_history("user", text)

        # 处理命令
        if text.startswith("/"):
            self._handle_command(text)
            return True

        # 运行 Agent
        self._run_agent(text)
        return True

    def _add_history(self, role: str, text: str) -> None:
        """添加历史消息并触发重绘."""
        self._history_entries.append((role, text))
        if len(self._history_entries) > 500:
            self._history_entries = self._history_entries[-250:]
        self.app.invalidate()

    def _handle_command(self, text: str) -> None:
        """处理内置命令."""
        parts = text.split()
        cmd = parts[0].lower()

        if cmd in ("/exit", "/quit"):
            self.app.exit()
            return

        if cmd == "/help":
            self._add_history("system", self._help_text())
            return

        if cmd == "/new":
            sid = self.sm.create()
            self._add_history("system", f"New session: {sid}")
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
                self._add_history(
                    "system",
                    f"Context: [{bar}] {pct}% ({used:,} / {limit:,} tokens)",
                )
            else:
                self._add_history("system", "Status not available.")
            return

        if cmd == "/clear":
            self._history_entries.clear()
            self.app.invalidate()
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
                self._add_history("system", "\n".join(lines))
            else:
                self._add_history("error", "No agent available.")
            return

        if cmd == "/history":
            agent = self.sm.get_current_agent()
            if agent:
                history = agent.get_history()
                if not history:
                    self._add_history("system", "History is empty.")
                    return
                lines = ["Conversation history:"]
                for i, msg in enumerate(history, 1):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if len(content) > 100:
                        content = content[:100] + "..."
                    lines.append(f"  {i}. [{role}] {content}")
                self._add_history("system", "\n".join(lines))
            else:
                self._add_history("error", "No agent available.")
            return

        if cmd == "/stream":
            agent = self.sm.get_current_agent()
            if agent:
                agent.streaming = not agent.streaming
                status = "on" if agent.streaming else "off"
                self._add_history("system", f"Streaming mode {status}.")
            return

        self._add_history(
            "error",
            f"Unknown command: {cmd}. Type /help for available commands.",
        )

    def _handle_session_command(self, parts: List[str]) -> None:
        """处理 /session 子命令."""
        if len(parts) < 2:
            self._add_history(
                "system",
                "Usage:\n"
                "  /session list\n"
                "  /session switch <ID>\n"
                "  /session rm <ID>\n"
                "  /session rename <ID> <NAME>",
            )
            return

        sub = parts[1].lower()
        if sub == "list":
            sessions = self.sm.list()
            if not sessions:
                self._add_history("system", "No sessions.")
                return
            lines = [f"Sessions ({len(sessions)}):"]
            for s in sessions:
                marker = "*" if s.get("is_current") else " "
                lines.append(
                    f"  [{marker}] {s['session_id']}  {s['name']}  "
                    f"({s.get('message_count', 0)} msgs)"
                )
            self._add_history("system", "\n".join(lines))
        elif sub == "switch":
            if len(parts) < 3:
                self._add_history("system", "Usage: /session switch <ID>")
                return
            sid = parts[2]
            if self.sm.switch(sid):
                self._add_history("system", f"Switched to: {sid}")
            else:
                self._add_history("error", f"Session not found: {sid}")
        elif sub in ("rm", "delete", "del"):
            if len(parts) < 3:
                self._add_history("system", "Usage: /session rm <ID>")
                return
            sid = parts[2]
            if self.sm.delete(sid):
                self._add_history("system", f"Deleted session: {sid}")
            else:
                self._add_history("error", f"Session not found: {sid}")
        elif sub == "rename":
            if len(parts) < 4:
                self._add_history("system", "Usage: /session rename <ID> <NAME>")
                return
            sid = parts[2]
            name = " ".join(parts[3:])
            if self.sm.rename(sid, name):
                self._add_history("system", f"Renamed to: {name}")
            else:
                self._add_history("error", f"Session not found: {sid}")
        else:
            self._add_history("error", f"Unknown subcommand: {sub}")

    def _run_agent(self, text: str) -> None:
        """在后台线程中运行 Agent."""
        self._status_text = "Thinking..."
        self.app.invalidate()

        def target():
            try:
                agent = self.sm.get_current_agent()
                if agent is None:
                    self._add_history("error", "No active session.")
                    return

                if self.verbose and hasattr(agent, "run_with_trace"):
                    self._run_agent_verbose(text)
                elif agent.streaming:
                    self._run_agent_stream(text)
                else:
                    result = agent.run(text)
                    self._add_history("assistant", result)

            except Exception as e:
                self._add_history("error", f"Agent error: {e}")
            finally:
                self._status_text = "Ready"
                self.app.invalidate()

        threading.Thread(target=target, daemon=True).start()

    def _run_agent_verbose(self, text: str) -> None:
        """以 verbose 模式运行 Agent."""
        agent = self.sm.get_current_agent()
        for event in agent.run_with_trace(text):
            etype = event.get("type")
            if etype == "thinking":
                self._add_history("thinking", event.get("text", ""))
            elif etype == "assistant":
                self._add_history("assistant", event.get("text", ""))
            elif etype == "tool_call":
                name = event.get("name", "")
                args = event.get("args", {})
                args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                self._add_history("tool", f"{name}({args_str})")
            elif etype == "observation":
                t = event.get("text", "")
                if len(t) > 200:
                    t = t[:200] + f" ... ({len(t)} chars)"
                self._add_history("system", t)
            elif etype == "error":
                self._add_history("error", event.get("text", ""))

    def _run_agent_stream(self, text: str) -> None:
        """以流式模式运行 Agent."""
        agent = self.sm.get_current_agent()
        chunks = []
        for chunk in agent.run_stream(text):
            chunks.append(chunk)
            combined = "".join(chunks)
            # 更新最后一条 assistant 消息（如果不存在则添加）
            if self._history_entries and self._history_entries[-1][0] == "assistant":
                self._history_entries[-1] = ("assistant", combined)
            else:
                self._history_entries.append(("assistant", combined))
            self.app.invalidate()

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
            "  /clear             — Clear history\n"
            "  /stream            — Toggle streaming mode\n"
            "  /exit              — Exit\n"
            "  (Any other text is sent to the Agent)"
        )

    def run(self) -> None:
        """启动应用."""
        self.app.run()


def run_tui(
    work_dir: str,
    auto_approve: bool = False,
    session_id: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """启动 TUI.

    优先使用 Application 固定布局；若终端不支持则回退到 REPL.
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

        Console().print(f"[red]Session not found: {session_id}[/red]")
        return

    try:
        app = AICodingApp(sm, verbose=verbose)
        app.run()
    except Exception as e:
        # 终端不支持 Application（如非 TTY），回退到 REPL
        from ai_coding.tui.render import console
        console.print(f"[dim]TUI not available ({e}), falling back to REPL.[/dim]")
        from ai_coding.tui.shell import run_shell
        run_shell(sm, verbose=verbose)
