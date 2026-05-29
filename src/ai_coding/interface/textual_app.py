"""Textual TUI 交互模块.

提供基于 Textual 的全终端交互界面，支持：
- 消息历史滚动
- 可折叠的 Thinking / Tool / Observation 面板（点击展开/收起）
- 流式输出实时显示
"""

from functools import partial
from typing import Any, Optional

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Collapsible, Input, Static

from ai_coding.logger import get_logger

logger = get_logger(__name__)


class UserMessage(Static):
    """用户消息组件."""

    def __init__(self, content: str, **kwargs: Any) -> None:
        super().__init__(f"[bold]>>>[/bold] {content}", **kwargs)
        self.add_class("user-message")


class AssistantMessage(Static):
    """AI 回复组件，支持动态追加内容."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._text = ""

    def append_text(self, text: str) -> None:
        self._text += text
        self.update(self._text)


class ThinkingCollapsible(Collapsible):
    """可折叠的思考过程面板."""

    def __init__(self, **kwargs: Any) -> None:
        self._label = Static("")
        super().__init__(
            self._label,
            title="[cyan]Thinking[/cyan]",
            collapsed=True,
            **kwargs,
        )
        self._text = ""

    def append_text(self, text: str) -> None:
        self._text += text
        self._label.update(self._text)


class ToolCallCollapsible(Collapsible):
    """可折叠的工具调用面板."""

    def __init__(self, name: str, args: dict, **kwargs: Any) -> None:
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        content = f"[yellow]{name}[/yellow]({args_str})"
        super().__init__(
            Static(content),
            title=f"[yellow]Tool: {name}[/yellow]",
            collapsed=True,
            **kwargs,
        )


class ReadFileCollapsible(Collapsible):
    """可折叠的观察结果面板."""

    def __init__(self, text: str, target: str = "", max_len: int = 200, **kwargs: Any) -> None:
        display = text[:max_len]
        if len(text) > max_len:
            display += f" ... ({len(text)} chars)"
        title = "[green]ReadFile[/green]"
        if target:
            title += f" [dim]{target}[/dim]"
        super().__init__(
            Static(display),
            title=title,
            collapsed=True,
            **kwargs,
        )


class ChatApp(App):
    """Textual 聊天应用主类."""

    CSS = """
    ChatApp {
        layout: vertical;
    }

    #chat-container {
        width: 100%;
        height: 1fr;
        padding: 0 1;
    }

    #input {
        dock: bottom;
        width: 100%;
        height: auto;
        margin: 0 1;
    }

    .welcome {
        margin: 0;
        text-style: bold;
        text-align: center;
    }

    .user-message {
        margin: 0;
        text-align: left;
        color: $text;
        border: solid gray;
        padding: 0 1;
    }

    AssistantMessage {
        margin: 0;
        text-align: left;
        color: $text;
    }

    ThinkingCollapsible {
        margin: 0;
    }

    ToolCallCollapsible {
        margin: 0;
    }

    ReadFileCollapsible {
        margin: 0;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(
        self,
        agent: Optional[Any] = None,
        verbose: bool = True,
        startup_info: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.agent = agent
        self.verbose = verbose
        self.startup_info = startup_info or {}
        self._current_thinking: Optional[ThinkingCollapsible] = None
        self._tool_calls: dict[str, dict] = {}

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-container")
        yield Input(
            placeholder="Type your message... (/help for commands, /exit to quit)",
            id="input",
        )

    def on_mount(self) -> None:
        """应用挂载后显示欢迎信息."""
        lines = ["AI Coding Assistant", "/help for commands  /exit to quit"]
        if self.startup_info:
            info = self.startup_info
            lines.append(f"Project: {info.get('project', 'N/A')}")
            lines.append(f"Provider: {info.get('provider', 'N/A')}")
            lines.append(f"Model: {info.get('model', 'N/A')}")
        self._mount_widget(
            Static(
                "\n".join(lines),
                classes="welcome",
            )
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """处理用户输入提交."""
        value = event.value.strip()
        if not value:
            return

        input_widget = self.query_one("#input", Input)
        input_widget.value = ""

        if value.startswith("/"):
            self._handle_command(value)
            return

        self._mount_widget(UserMessage(value))
        self.run_worker(
            partial(self._process_agent_response, value),
            thread=True,
        )

    def _mount_widget(self, widget: Static) -> None:
        """将 widget 挂载到聊天容器并滚动到底部."""
        container = self.query_one("#chat-container", VerticalScroll)
        container.mount(widget)
        container.scroll_end()

    def _process_agent_response(self, user_input: str) -> None:
        """在线程中处理 Agent 响应."""
        if not self.agent:
            self.call_from_thread(
                self._mount_widget,
                AssistantMessage(),
            )
            return

        logger.info(f"User input: {user_input}")

        if self.verbose and hasattr(self.agent, "run_with_trace"):
            self._process_trace(user_input)
        elif getattr(self.agent, "streaming", False):
            self._process_stream(user_input)
        else:
            result = self.agent.run(user_input)
            msg = AssistantMessage()
            self.call_from_thread(self._mount_widget, msg)
            self.call_from_thread(msg.append_text, result)

    def _process_trace(self, user_input: str) -> None:
        """处理带轨迹的输出（verbose 模式）."""
        self._current_thinking = None
        self._tool_calls.clear()
        assistant_msg: Optional[AssistantMessage] = None

        for event in self.agent.run_with_trace(user_input):
            etype = event.get("type")
            if etype in ("tool_call", "observation", "error"):
                self._current_thinking = None

            if etype == "thinking":
                if self._current_thinking is None:
                    panel = ThinkingCollapsible()
                    self._current_thinking = panel
                    self.call_from_thread(self._mount_widget, panel)
                self.call_from_thread(
                    self._current_thinking.append_text,
                    event["text"],
                )

            elif etype == "assistant":
                if assistant_msg is None:
                    assistant_msg = AssistantMessage()
                    self.call_from_thread(self._mount_widget, assistant_msg)
                self.call_from_thread(
                    assistant_msg.append_text,
                    event["text"],
                )

            elif etype == "tool_call":
                panel = ToolCallCollapsible(
                    event["name"],
                    event["args"],
                )
                self.call_from_thread(self._mount_widget, panel)
                if event.get("id"):
                    self._tool_calls[event["id"]] = {
                        "name": event["name"],
                        "args": event["args"],
                    }

            elif etype == "observation":
                tool_call_id = event.get("tool_call_id")
                target = ""
                if tool_call_id and tool_call_id in self._tool_calls:
                    tc = self._tool_calls[tool_call_id]
                    if tc.get("name") == "read_file":
                        target = tc.get("args", {}).get("path", "")
                panel = ReadFileCollapsible(event["text"], target=target)
                self.call_from_thread(self._mount_widget, panel)

            elif etype == "error":
                self.call_from_thread(
                    self._mount_widget,
                    Static(f"[red]Error: {event['text']}[/red]"),
                )

    def _process_stream(self, user_input: str) -> None:
        """处理流式输出."""
        msg = AssistantMessage()
        self.call_from_thread(self._mount_widget, msg)

        for chunk in self.agent.run_stream(user_input):
            self.call_from_thread(msg.append_text, chunk)

    def _handle_command(self, command: str) -> None:
        """处理内置命令."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        handlers = {
            "/exit": self._cmd_exit,
            "/quit": self._cmd_exit,
            "/help": self._cmd_help,
            "/clear": self._cmd_clear,
            "/status": self._cmd_status,
            "/tools": self._cmd_tools,
            "/history": self._cmd_history,
            "/verbose": self._cmd_verbose,
            "/stream": self._cmd_stream,
        }

        handler = handlers.get(cmd, self._cmd_unknown)
        result = handler(args)
        if result:
            self._mount_widget(Static(f"[dim]{result}[/dim]"))

    def _cmd_exit(self, args: str) -> str:
        self.exit()
        return ""

    def _cmd_help(self, args: str) -> str:
        return (
            "Available commands:\n"
            "  /help     - Show help\n"
            "  /status   - Show context usage\n"
            "  /tools    - List available tools\n"
            "  /history  - Show conversation history\n"
            "  /verbose  - Toggle verbose mode\n"
            "  /stream   - Toggle streaming mode\n"
            "  /clear    - Clear screen\n"
            "  /exit     - Exit"
        )

    def _cmd_status(self, args: str) -> str:
        if not self.agent:
            return "Agent not initialized."
        if not hasattr(self.agent, "get_context_usage"):
            return "Status not available."
        usage = self.agent.get_context_usage()
        used = usage.get("used_tokens", 0)
        limit = usage.get("limit_tokens", 128000)
        pct = usage.get("percentage", 0.0)
        bar_len = 20
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        return f"Context: [{bar}] {pct}% ({used:,} / {limit:,} tokens)"

    def _cmd_clear(self, args: str) -> str:
        container = self.query_one("#chat-container", VerticalScroll)
        container.remove_children()
        if self.agent and hasattr(self.agent, "clear_history"):
            self.agent.clear_history()
        return "Session cleared."

    def _cmd_tools(self, args: str) -> str:
        if not self.agent:
            return "Agent not initialized."
        stats = self.agent.get_stats()
        tools = stats.get("tools", [])
        if not tools:
            return "No tools available."
        lines = [f"Available tools ({len(tools)}):"]
        for name in tools:
            lines.append(f"  - {name}")
        return "\n".join(lines)

    def _cmd_history(self, args: str) -> str:
        if not self.agent:
            return "Agent not initialized."
        history = self.agent.get_history()
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

    def _cmd_verbose(self, args: str) -> str:
        self.verbose = not self.verbose
        status = "on" if self.verbose else "off"
        return f"Verbose mode {status}."

    def _cmd_stream(self, args: str) -> str:
        if self.agent:
            self.agent.streaming = not self.agent.streaming
            status = "on" if self.agent.streaming else "off"
            return f"Streaming mode {status}."
        return "Agent not initialized."

    def _cmd_unknown(self, args: str) -> str:
        return "Unknown command. Type /help for available commands."
