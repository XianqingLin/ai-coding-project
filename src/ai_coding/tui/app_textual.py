"""AI Coding TUI 应用 —— Textual 全屏模式.

特点：
- 全屏固定布局（可滚动历史 + 边框输入框 + 状态栏）
- 组件级差分渲染，无闪烁
- 后台线程运行 Agent，通过 call_from_thread 安全更新 UI
- 支持流式输出实时更新
- thinking 和工具调用以 Collapsible 折叠面板展示
"""

import os
from typing import List, Optional

from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Input, Static, Collapsible

from ai_coding.agent import SessionManager
from ai_coding.agent.core import DEFAULT_CONTEXT_LIMIT
from ai_coding.config import DEFAULT_LLM_PROVIDER


class AICodingApp(App):
    """AI Coding TUI 主应用."""

    CSS = """
    Screen {
        align: center middle;
    }

    #history {
        width: 100%;
        height: 1fr;
        border: none;
        padding: 0 1;
    }

    #input {
        width: 100%;
        height: auto;
        min-height: 1;
        border: solid #4a90e2;
        padding: 0 1;
    }

    #statusbar {
        width: 100%;
        height: 1;
        background: #1e1e2e;
        color: #6c7086;
        content-align: left middle;
        padding: 0 1;
    }

    .user-message {
        color: #f9e2af;
        text-style: bold;
        margin: 1 0;
    }

    .assistant-message {
        color: #cdd6f4;
        margin: 0 0 1 0;
    }

    .system-message {
        color: #a6e3a1;
        margin: 0 0;
    }

    .error-message {
        color: #f38ba8;
        margin: 0 0;
    }

    /* ---------- Collapsible 面板 ---------- */

    .thinking-collapsible {
        border: solid #6c7086;
        margin: 1 0;
        padding: 0;
        background: transparent;
    }

    .thinking-collapsible CollapsibleTitle {
        color: #6c7086;
        text-style: italic;
        background: transparent;
    }

    .thinking-collapsible CollapsibleContents {
        color: #6c7086;
        text-style: italic;
        padding: 0 1;
        background: transparent;
    }

    .tool-collapsible {
        border: solid #a6e3a1;
        margin: 1 0;
        padding: 0;
        background: transparent;
    }

    .tool-collapsible CollapsibleTitle {
        color: #a6e3a1;
        background: transparent;
    }

    .tool-collapsible CollapsibleContents {
        color: #cdd6f4;
        padding: 0 1;
        background: transparent;
    }

    .tool-collapsible-done CollapsibleTitle {
        color: #89b4fa;
        background: transparent;
    }
    """

    def __init__(self, sm: SessionManager) -> None:
        self.sm = sm
        self._pending_input: str = ""
        self._last_assistant_widget: Optional[Static] = None
        self._last_thinking_widget: Optional[Static] = None
        self._last_tool_widget: Optional[Static] = None
        self._last_tool_collapsible: Optional[Collapsible] = None
        self._current_assistant_text: str = ""
        self._current_thinking_text: str = ""
        super().__init__()

    # ------------------------------------------------------------------ #
    # 布局
    # ------------------------------------------------------------------ #

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="history"):
            yield Static(id="welcome")
        yield Input(placeholder="Enter command...", id="input")
        yield Static(id="statusbar")

    def on_mount(self) -> None:
        """初始化界面."""
        self._render_welcome()
        self._update_statusbar()
        self.set_interval(1.0, self._update_statusbar)
        self.query_one("#input", Input).focus()

    # ------------------------------------------------------------------ #
    # 内容渲染
    # ------------------------------------------------------------------ #

    def _render_welcome(self) -> None:
        """渲染 Welcome Panel."""
        welcome = self.query_one("#welcome", Static)
        session_name = self.sm.current.name if self.sm.current else "default"
        lines = [
            "[bold #89b4fa]Welcome to AI Coding![/bold #89b4fa]",
            "[dim]Send /help for help information.[/dim]",
            "",
            f"[dim]Directory:[/dim] {self.sm.work_dir}",
            f"[dim]Session:[/dim]   {session_name}",
            f"[dim]Model:[/dim]     {DEFAULT_LLM_PROVIDER}",
        ]
        panel = Panel("\n".join(lines), border_style="#4a90e2", padding=(0, 1))
        welcome.update(panel)

    def _update_statusbar(self) -> None:
        """更新状态栏."""
        statusbar = self.query_one("#statusbar", Static)
        agent = self.sm.get_current_agent()
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

        session = self.sm.current
        session_info = ""
        if session:
            session_info = f"{session.name} ({session.session_id})"

        left = f"{DEFAULT_LLM_PROVIDER}  {session_info}  {self.sm.work_dir}"
        right = context_info

        try:
            width = self.size.width
        except Exception:
            width = 80

        padding = max(width - len(left) - len(right), 1)
        statusbar.update(f"{left}{' ' * padding}{right}")

    # ------------------------------------------------------------------ #
    # 消息操作
    # ------------------------------------------------------------------ #

    def _add_user_message(self, text: str) -> None:
        """添加用户消息."""
        history = self.query_one("#history", VerticalScroll)
        msg = Static(f"✨ {text}", classes="user-message")
        history.mount(msg)
        history.scroll_end(animate=False)

    def _add_assistant_message(self, text: str) -> Static:
        """添加助手消息."""
        history = self.query_one("#history", VerticalScroll)
        msg = Static(f"● {text}", classes="assistant-message")
        history.mount(msg)
        history.scroll_end(animate=False)
        return msg

    def _start_new_assistant_response(self) -> None:
        """为新的 assistant 流式回复创建空占位符."""
        self._current_assistant_text = ""
        self._last_assistant_widget = self._add_assistant_message("")

    def _append_assistant_chunk(self, text: str) -> None:
        """追加 assistant chunk 到当前占位符（流式中用纯文本）."""
        if self._last_assistant_widget is None:
            self._start_new_assistant_response()
        self._current_assistant_text += text
        # 流式过程中显示原始文本+进度指示，避免未闭合 Markdown 标记导致格式错乱
        self._last_assistant_widget.update(f"● {self._current_assistant_text}")
        history = self.query_one("#history", VerticalScroll)
        history.scroll_end(animate=False)

    def _finish_assistant_response(self) -> None:
        """Assistant 流式结束，用 Markdown 渲染最终回复."""
        if self._last_assistant_widget is None:
            return
        md = RichMarkdown(self._current_assistant_text)
        self._last_assistant_widget.update(md)

    # ------------------------------------------------------------------ #
    # Thinking 折叠面板
    # ------------------------------------------------------------------ #

    def _start_new_thinking_response(self) -> None:
        """为新的 thinking 流式回复创建 Collapsible 占位符."""
        self._current_thinking_text = ""
        history = self.query_one("#history", VerticalScroll)
        content = Static("", classes="thinking-collapsible-content")
        collapsible = Collapsible(
            content,
            title="⚡ Thinking",
            collapsed=True,
            classes="thinking-collapsible",
        )
        history.mount(collapsible)
        history.scroll_end(animate=False)
        self._last_thinking_widget = content

    def _append_thinking_chunk(self, text: str) -> None:
        """追加 thinking chunk 到当前折叠面板."""
        if self._last_thinking_widget is None:
            self._start_new_thinking_response()
        self._current_thinking_text += text
        self._last_thinking_widget.update(self._current_thinking_text)
        history = self.query_one("#history", VerticalScroll)
        history.scroll_end(animate=False)

    # ------------------------------------------------------------------ #
    # Tool 折叠面板
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_tool_title(name: str, args: dict) -> str:
        """根据工具名和参数生成折叠面板标题."""
        if not args:
            return f"● {name}"

        # list_dir：显示为 "list dir <目标目录>"
        if name == "list_dir" and "path" in args:
            path = os.path.basename(str(args["path"])) or str(args["path"])
            if len(path) > 30:
                path = path[:27] + "..."
            return f"● list dir  {path}"

        # 文件类工具：显示目标文件名（更简洁）
        if "path" in args:
            path = os.path.basename(str(args["path"])) or str(args["path"])
            if len(path) > 30:
                path = path[:27] + "..."
            return f"● {name}  {path}"

        # Shell：截断长命令
        if name == "Shell" and "command" in args:
            cmd = str(args["command"]).strip()
            if len(cmd) > 40:
                cmd = cmd[:40] + "..."
            return f"● {name}  {cmd}"

        # SetTodoList：固定描述
        if name == "SetTodoList":
            return "● SetTodoList  Update Todos"

        # 通用处理：取第一个参数值
        first_val = str(next(iter(args.values())))
        if len(first_val) > 40:
            first_val = first_val[:40] + "..."
        return f"● {name}  {first_val}"

    def _add_tool_call(self, name: str, args: dict) -> None:
        """添加工具调用折叠面板."""
        title = self._format_tool_title(name, args)

        history = self.query_one("#history", VerticalScroll)
        content = Static("[dim]Running...[/dim]", classes="tool-collapsible-content")
        collapsible = Collapsible(
            content,
            title=title,
            collapsed=True,
            classes="tool-collapsible",
        )
        history.mount(collapsible)
        history.scroll_end(animate=False)
        self._last_tool_widget = content
        self._last_tool_collapsible = collapsible

    def _add_observation(self, text: str) -> None:
        """添加工具观察结果到最近一个工具面板."""
        if self._last_tool_widget is None:
            self._add_system_message(f"[Observation] {text}")
            return

        self._last_tool_widget.update(text)

        # 工具执行完成，把标题圆点改为对勾，并切换样式
        if self._last_tool_collapsible is not None:
            try:
                old = self._last_tool_collapsible.title
                self._last_tool_collapsible.title = old.replace("● ", "✓ ", 1)
                self._last_tool_collapsible.add_class("tool-collapsible-done")
            except Exception:
                pass

        history = self.query_one("#history", VerticalScroll)
        history.scroll_end(animate=False)

    def _add_system_message(self, text: str) -> None:
        """添加系统消息."""
        history = self.query_one("#history", VerticalScroll)
        msg = Static(text, classes="system-message")
        history.mount(msg)
        history.scroll_end(animate=False)

    def _add_error_message(self, text: str) -> None:
        """添加错误消息."""
        history = self.query_one("#history", VerticalScroll)
        msg = Static(text, classes="error-message")
        history.mount(msg)
        history.scroll_end(animate=False)

    def _clear_history(self) -> None:
        """清空历史（保留 Welcome）."""
        history = self.query_one("#history", VerticalScroll)
        for child in list(history.children):
            if child.id != "welcome":
                child.remove()

    # ------------------------------------------------------------------ #
    # 交互逻辑
    # ------------------------------------------------------------------ #

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """处理用户输入."""
        text = event.value.strip()
        if not text:
            return

        # 清空输入框
        input_widget = self.query_one("#input", Input)
        input_widget.value = ""

        # 显示用户消息
        self._add_user_message(text)

        # 处理命令
        if text.startswith("/"):
            self._handle_command(text)
            return

        # 后台运行 Agent
        self._pending_input = text
        self.run_worker(self._run_agent_task, thread=True, exclusive=True)

    # ------------------------------------------------------------------ #
    # 命令处理
    # ------------------------------------------------------------------ #

    def _handle_command(self, text: str) -> None:
        """处理内置命令."""
        parts = text.split()
        cmd = parts[0].lower()
        agent = self.sm.get_current_agent()

        if cmd in ("/exit", "/quit"):
            self.exit()
            return

        if cmd == "/help":
            self._add_system_message(
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
                "  /exit              — Exit\n"
                "  (Any other text is sent to the Agent)"
            )
            return

        if cmd == "/status":
            if agent is None or not hasattr(agent, "get_context_usage"):
                self._add_system_message("Status not available.")
                return
            usage = agent.get_context_usage()
            used = usage.get("used_tokens", 0)
            limit = usage.get("limit_tokens", DEFAULT_CONTEXT_LIMIT)
            pct = usage.get("percentage", 0.0)
            bar_len = 20
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            self._add_system_message(
                f"Context: [{bar}] {pct}% ({used:,} / {limit:,} tokens)"
            )
            return

        if cmd == "/clear":
            self._clear_history()
            return

        if cmd == "/new":
            sid = self.sm.create()
            session = self.sm.current
            self._add_system_message(
                f"New session: {session.name if session else sid}"
            )
            self._render_welcome()
            self._update_statusbar()
            return

        if cmd == "/tools":
            if agent is None:
                self._add_error_message("No agent available.")
                return
            stats = agent.get_stats()
            tools = stats.get("tools", [])
            if not tools:
                self._add_system_message("No tools available.")
                return
            lines = [f"Available tools ({len(tools)}):"]
            for name in tools:
                lines.append(f"  - {name}")
            self._add_system_message("\n".join(lines))
            return

        if cmd == "/history":
            if agent is None:
                self._add_error_message("No agent available.")
                return
            history = agent.get_history()
            if not history:
                self._add_system_message("History is empty.")
                return
            lines = ["Conversation history:"]
            for i, msg in enumerate(history, 1):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if len(content) > 100:
                    content = content[:100] + "..."
                lines.append(f"  {i}. [{role}] {content}")
            self._add_system_message("\n".join(lines))
            return

        if cmd == "/session":
            self._handle_session_command(parts)
            return

        self._add_error_message(
            f"Unknown command: {cmd}. Type /help for available commands."
        )

    def _handle_session_command(self, parts: List[str]) -> None:
        """处理 /session 子命令."""
        if len(parts) < 2:
            self._add_system_message(
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
                self._add_system_message("No sessions.")
                return
            lines = [f"Sessions ({len(sessions)}):"]
            for s in sessions:
                marker = "*" if s.get("is_current") else " "
                lines.append(
                    f"  [{marker}] {s['session_id']}  {s['name']}  "
                    f"({s.get('message_count', 0)} msgs)"
                )
            self._add_system_message("\n".join(lines))

        elif sub == "switch":
            if len(parts) < 3:
                self._add_system_message("Usage: /session switch <ID>")
                return
            sid = parts[2]
            if self.sm.switch(sid):
                self._add_system_message(f"Switched to: {sid}")
                self._render_welcome()
                self._update_statusbar()
            else:
                self._add_error_message(f"Session not found: {sid}")

        elif sub in ("rm", "delete", "del"):
            if len(parts) < 3:
                self._add_system_message("Usage: /session rm <ID>")
                return
            sid = parts[2]
            if self.sm.delete(sid):
                current = self.sm.current
                if current and current.session_id != sid:
                    self._add_system_message(
                        f"Deleted session: {sid}. "
                        f"Current session: {current.name} ({current.session_id})"
                    )
                else:
                    self._add_system_message(f"Deleted session: {sid}")
                self._render_welcome()
                self._update_statusbar()
            else:
                self._add_error_message(f"Session not found: {sid}")

        elif sub == "rename":
            if len(parts) < 4:
                self._add_system_message("Usage: /session rename <ID> <NAME>")
                return
            sid = parts[2]
            name = " ".join(parts[3:])
            if self.sm.rename(sid, name):
                self._add_system_message(f"Renamed to: {name}")
                self._update_statusbar()
            else:
                self._add_error_message(f"Session not found: {sid}")

        else:
            self._add_error_message(f"Unknown subcommand: {sub}")

    # ------------------------------------------------------------------ #
    # Agent 运行
    # ------------------------------------------------------------------ #

    def _run_agent_task(self) -> None:
        """在后台线程中运行 Agent（由 run_worker 调用).

        统一使用流式+verbose 模式.
        """
        user_input = self._pending_input
        self._last_assistant_widget = None
        self._last_thinking_widget = None
        self._last_tool_widget = None
        self._last_tool_collapsible = None
        self._current_assistant_text = ""
        self._current_thinking_text = ""
        agent = self.sm.get_current_agent()

        if agent is None:
            self.call_from_thread(self._add_error_message, "No active session.")
            return

        if not hasattr(agent, "run_stream_verbose"):
            self.call_from_thread(
                self._add_error_message,
                "Agent does not support stream+verbose mode.",
            )
            return

        try:
            for event in agent.run_stream_verbose(user_input):
                etype = event.get("type")
                if etype == "thinking_start":
                    self.call_from_thread(self._start_new_thinking_response)
                elif etype == "thinking_chunk":
                    self.call_from_thread(
                        self._append_thinking_chunk, event.get("text", "")
                    )
                elif etype == "assistant_start":
                    self.call_from_thread(self._start_new_assistant_response)
                elif etype == "assistant_chunk":
                    self.call_from_thread(
                        self._append_assistant_chunk, event.get("text", "")
                    )
                elif etype == "assistant_end":
                    self.call_from_thread(self._finish_assistant_response)
                elif etype == "tool_call":
                    self.call_from_thread(
                        self._add_tool_call,
                        event.get("name", ""),
                        event.get("args", {}),
                    )
                elif etype == "observation":
                    self.call_from_thread(
                        self._add_observation, event.get("text", "")
                    )
                elif etype == "error":
                    self.call_from_thread(
                        self._add_error_message, event.get("text", "")
                    )
        except Exception as e:
            self.call_from_thread(self._add_error_message, f"Agent error: {e}")
