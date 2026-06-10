"""AI Coding TUI 自定义组件（Claude Code 风格）."""

import re
from enum import Enum
from typing import Optional

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Input, Label, Static


class MessageType(Enum):
    """消息类型."""

    USER = "user"
    AI = "ai"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"
    ERROR = "error"


CODE_BLOCK_RE = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)


def render_rich_content(content: str) -> RenderableType:
    """解析文本为 Rich Renderable，支持 Markdown 代码块语法高亮."""
    if "```" not in content:
        return Text(content)

    parts: list[RenderableType] = []
    last_end = 0

    for match in CODE_BLOCK_RE.finditer(content):
        # 代码块前的文本
        if match.start() > last_end:
            text_part = content[last_end : match.start()].strip()
            if text_part:
                parts.append(Text(text_part))

        # 提取代码块
        lang = match.group(1) or "text"
        code = match.group(2).rstrip()
        try:
            syntax = Syntax(code, lang, theme="monokai", line_numbers=False)
            panel = Panel(syntax, border_style="#6495ED", padding=(0, 1))
            parts.append(panel)
        except Exception:
            parts.append(Panel(code, border_style="blue", padding=(0, 1)))

        last_end = match.end()

    # 剩余文本
    if last_end < len(content):
        text_part = content[last_end:].strip()
        if text_part:
            parts.append(Text(text_part))

    return Group(*parts) if len(parts) > 1 else (parts[0] if parts else Text(""))


class MessageBubble(Widget):
    """消息气泡组件，不同角色有不同视觉样式."""

    DEFAULT_CSS = """
    MessageBubble {
        width: 100%;
        margin: 1 0;
    }
    .bubble-inner {
        width: 100%;
    }
    .msg-user .bubble-inner {
        background: $primary-darken-3;
        border-left: solid $primary;
    }
    .msg-ai .bubble-inner {
        background: $surface;
        border-left: solid $success;
    }
    .msg-tool_call .bubble-inner {
        background: $warning-darken-3;
        border-left: solid $warning;
    }
    .msg-error .bubble-inner {
        background: $error-darken-3;
        border-left: solid $error;
    }
    .msg-system .bubble-inner {
        background: $surface-darken-1;
        border-left: solid $foreground;
    }
    .bubble-header {
        height: 1;
        text-style: bold;
        padding: 0 1;
        color: $foreground;
    }
    .bubble-content {
        padding: 0 1 1 1;
    }
    """

    _LABEL_MAP = {
        MessageType.USER: "You",
        MessageType.AI: "Assistant",
        MessageType.TOOL_CALL: "Tool Call",
        MessageType.TOOL_RESULT: "Tool Result",
        MessageType.ERROR: "Error",
        MessageType.SYSTEM: "System",
    }

    def __init__(
        self,
        content: str,
        msg_type: MessageType = MessageType.AI,
        **kwargs,
    ):
        self.msg_type = msg_type
        self.raw_content = content
        super().__init__(**kwargs)
        self.add_class(f"msg-{msg_type.value}")

    def compose(self) -> ComposeResult:
        with Vertical(classes="bubble-inner"):
            yield Label(self._LABEL_MAP.get(self.msg_type, "Info"), classes="bubble-header")
            yield Static(classes="bubble-content")

    def on_mount(self) -> None:
        content_widget = self.query_one(".bubble-content", Static)
        renderable = render_rich_content(self.raw_content)
        content_widget.update(renderable)


class ChatLog(VerticalScroll):
    """聊天日志区，支持滚动和消息追加."""

    DEFAULT_CSS = """
    ChatLog {
        height: 1fr;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Vertical(id="chat-container")

    def add_message(self, content: str, msg_type: MessageType) -> None:
        """添加消息并自动滚动到底部."""
        bubble = MessageBubble(content, msg_type)
        container = self.query_one("#chat-container", Vertical)
        container.mount(bubble)
        self.scroll_end()

    def clear_messages(self) -> None:
        """清空所有消息."""
        container = self.query_one("#chat-container", Vertical)
        container.remove_children()


class InputBar(Horizontal):
    """底部输入栏，支持 Enter 发送和按钮发送."""

    class Submitted(Message):
        """自定义提交消息."""

        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    DEFAULT_CSS = """
    InputBar {
        height: auto;
        max-height: 6;
        background: $surface;
        border-top: solid $primary-darken-2;
        padding: 0 1;
    }
    #prompt-input {
        width: 1fr;
    }
    #send-button {
        width: 8;
        margin-left: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Input(placeholder=">>> 输入指令 (Enter 发送)", id="prompt-input")
        yield Button("发送", id="send-button", variant="primary")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.post_message(self.Submitted(event.value))
        self.query_one("#prompt-input", Input).value = ""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send-button":
            input_widget = self.query_one("#prompt-input", Input)
            self.post_message(self.Submitted(input_widget.value))
            input_widget.value = ""


class StatusBar(Horizontal):
    """顶部状态栏."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        background: $primary-darken-2;
        color: $foreground;
        padding: 0 2;
    }
    .status-brand {
        width: 20%;
        text-style: bold;
    }
    .status-model {
        width: 30%;
        content-align: center middle;
    }
    .status-state {
        width: 25%;
        content-align: right middle;
    }
    .status-tokens {
        width: 25%;
        content-align: right middle;
    }
    """

    state_text: reactive[str] = reactive("Ready")

    def compose(self) -> ComposeResult:
        yield Label("AI Coding", classes="status-brand")
        yield Label("", classes="status-model")
        yield Label("", classes="status-state")
        yield Label("", classes="status-tokens")

    def watch_state_text(self, state: str) -> None:
        self.query_one(".status-state", Label).update(state)

    def set_model(self, model: str) -> None:
        self.query_one(".status-model", Label).update(model)

    def set_tokens(self, tokens: str) -> None:
        self.query_one(".status-tokens", Label).update(tokens)


class ApprovalModal(ModalScreen[bool]):
    """工具调用审批弹窗."""

    DEFAULT_CSS = """
    ApprovalModal {
        align: center middle;
    }
    #modal-box {
        width: 60;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }
    .modal-title {
        text-style: bold;
        text-align: center;
        height: 1;
        margin-bottom: 1;
    }
    .modal-args {
        max-height: 20;
        overflow-y: auto;
        border: solid $primary-darken-2;
        padding: 1;
        margin: 1 0;
    }
    .modal-buttons {
        height: auto;
        align: center middle;
        margin-top: 1;
    }
    #btn-yes {
        margin-right: 2;
    }
    """

    def __init__(self, tool_name: str, args: dict, **kwargs):
        self.tool_name = tool_name
        self.args = args
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Label("Approve Tool Call?", classes="modal-title")
            yield Label(f"Tool: {self.tool_name}")
            yield Static(str(self.args), classes="modal-args")
            with Horizontal(classes="modal-buttons"):
                yield Button("Yes", id="btn-yes", variant="success")
                yield Button("No", id="btn-no", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-yes":
            self.dismiss(True)
        else:
            self.dismiss(False)
