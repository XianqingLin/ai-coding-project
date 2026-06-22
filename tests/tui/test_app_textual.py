"""Textual TUI 单元测试.

使用 Textual 内置的 Pilot 测试框架在 headless 模式下验证界面组件、
命令处理和事件渲染.
"""

from typing import Any, Dict, Iterator, List, Optional

import pytest
from textual.containers import VerticalScroll
from textual.pilot import Pilot
from textual.widgets import Collapsible, Input, Static

from ai_coding.agent.events import (
    AgentEvent,
    AssistantChunkEvent,
    AssistantEndEvent,
    AssistantStartEvent,
    ErrorEvent,
    ObservationEvent,
    ThinkingChunkEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallEvent,
)
from ai_coding.tools.base import ToolResult
from ai_coding.tui.app_textual import AICodingApp

pytestmark = pytest.mark.anyio


class _MockAgentService:
    """用于 TUI 测试的 Mock AgentService."""

    def __init__(self, work_dir: str = "/tmp/test") -> None:
        self.work_dir = work_dir
        self._current = {
            "session_id": "sess_001",
            "name": "default",
            "message_count": 0,
        }
        self._sessions = [self._current]
        self._events: List[AgentEvent] = []
        self.last_input: str = ""
        self.compact_called: bool = False

    @property
    def current_session_id(self) -> Optional[str]:
        return self._current["session_id"]

    def create_session(self, name: str = "") -> str:
        sid = f"sess_{len(self._sessions) + 1:03d}"
        self._sessions.append(
            {
                "session_id": sid,
                "name": name or sid,
                "message_count": 0,
            }
        )
        self._current = self._sessions[-1]
        return sid

    def list_sessions(self) -> List[Dict[str, Any]]:
        sessions = []
        for s in self._sessions:
            sessions.append(
                {
                    **s,
                    "is_current": s["session_id"] == self._current["session_id"],
                }
            )
        return sessions

    def switch_session(self, session_id: str) -> bool:
        for s in self._sessions:
            if s["session_id"] == session_id:
                self._current = s
                return True
        return False

    def delete_session(self, session_id: str) -> bool:
        original_len = len(self._sessions)
        self._sessions = [s for s in self._sessions if s["session_id"] != session_id]
        return len(self._sessions) < original_len

    def rename_session(self, session_id: str, new_name: str) -> bool:
        for s in self._sessions:
            if s["session_id"] == session_id:
                s["name"] = new_name
                return True
        return False

    def get_current_session(self) -> Optional[Dict[str, Any]]:
        return self._current

    def get_history(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return []

    def get_context_usage(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        return {"used_tokens": 100, "limit_tokens": 10000, "percentage": 1.0}

    def get_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        return {}

    def get_system_prompt(self, session_id: Optional[str] = None) -> str:
        return "You are an AI coding assistant."

    def send_message_stream(
        self, text: str, session_id: Optional[str] = None
    ) -> Iterator[AgentEvent]:
        """根据预设事件流返回."""
        self.last_input = text
        for event in self._events:
            yield event

    def set_events(self, events: List[AgentEvent]) -> None:
        self._events = events

    def compact_memory(self, session_id: Optional[str] = None):
        """模拟记忆压缩."""
        self.compact_called = True
        return ToolResult.ok("Compacted 2 memories")


@pytest.fixture
def mock_service() -> _MockAgentService:
    return _MockAgentService()


async def _type_string(pilot: Pilot, text: str) -> None:
    """通过逐字符 press 模拟输入字符串."""
    for char in text:
        await pilot.press(char)


def _static_text(widget: Static) -> str:
    """获取 Static widget 的文本内容（支持 Rich renderable）."""
    from rich.console import Console

    content = widget._Static__content  # type: ignore[misc]
    if isinstance(content, str):
        return content
    console = Console(record=True, force_terminal=False, width=80)
    console.print(content)
    return console.export_text()


class TestAICodingAppCompose:
    """界面组件初始化测试."""

    async def test_compose_creates_widgets(
        self, mock_service: _MockAgentService
    ) -> None:
        """compose() 应创建 history、input、statusbar."""
        app = AICodingApp(mock_service)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert len(app.query("#history")) == 1
            assert len(app.query("#input")) == 1
            assert len(app.query("#statusbar")) == 1
            assert isinstance(app.query_one("#history"), VerticalScroll)
            assert isinstance(app.query_one("#input"), Input)
            assert isinstance(app.query_one("#statusbar"), Static)

    async def test_welcome_renders_session_info(
        self, mock_service: _MockAgentService
    ) -> None:
        """Welcome 面板应显示工作目录和会话名."""
        app = AICodingApp(mock_service)
        async with app.run_test() as pilot:
            await pilot.pause()
            welcome = app.query_one("#welcome", Static)
            text = _static_text(welcome)
            assert "AI Coding" in text
            assert mock_service.work_dir in text
            assert "default" in text


class TestAICodingAppCommands:
    """内置命令测试."""

    async def test_help_command(self, mock_service: _MockAgentService) -> None:
        """/help 应显示帮助信息."""
        app = AICodingApp(mock_service)
        async with app.run_test() as pilot:
            await _type_string(pilot, "/help")
            await pilot.press("enter")
            await pilot.pause()

            history = app.query_one("#history", VerticalScroll)
            messages = list(history.query(".system-message"))
            assert len(messages) >= 1
            assert "Available commands" in _static_text(messages[-1])

    async def test_clear_command(self, mock_service: _MockAgentService) -> None:
        """/clear 应清空历史但保留 welcome."""
        app = AICodingApp(mock_service)
        async with app.run_test() as pilot:
            await _type_string(pilot, "/help")
            await pilot.press("enter")
            await pilot.pause()

            history = app.query_one("#history", VerticalScroll)
            before_count = len(list(history.children))
            assert before_count > 1

            await _type_string(pilot, "/clear")
            await pilot.press("enter")
            await pilot.pause()

            after_count = len(list(history.children))
            assert after_count == 1
            assert history.query_one("#welcome") is not None

    async def test_new_command(self, mock_service: _MockAgentService) -> None:
        """/new 应创建新会话并更新 welcome."""
        app = AICodingApp(mock_service)
        async with app.run_test() as pilot:
            await _type_string(pilot, "/new")
            await pilot.press("enter")
            await pilot.pause()

            assert len(mock_service._sessions) == 2
            welcome = app.query_one("#welcome", Static)
            assert "sess_002" in _static_text(welcome)

    async def test_session_list_command(self, mock_service: _MockAgentService) -> None:
        """/session list 应列出会话."""
        app = AICodingApp(mock_service)
        async with app.run_test() as pilot:
            await _type_string(pilot, "/session list")
            await pilot.press("enter")
            await pilot.pause()

            history = app.query_one("#history", VerticalScroll)
            messages = list(history.query(".system-message"))
            assert len(messages) >= 1
            assert "Sessions" in _static_text(messages[-1])

    async def test_exit_command(self, mock_service: _MockAgentService) -> None:
        """/exit 应退出应用."""
        app = AICodingApp(mock_service)
        async with app.run_test() as pilot:
            await _type_string(pilot, "/exit")
            await pilot.press("enter")
            await pilot.pause()

            assert app._exit
            assert mock_service.compact_called is False

    async def test_compact_command(self, mock_service: _MockAgentService) -> None:
        """/compact 应显示压缩结果."""
        app = AICodingApp(mock_service)
        async with app.run_test() as pilot:
            await _type_string(pilot, "/compact")
            await pilot.press("enter")
            await pilot.pause()

            history = app.query_one("#history", VerticalScroll)
            messages = list(history.query(".system-message"))
            assert len(messages) >= 1
            assert "Compacted 2 memories" in _static_text(messages[-1])
            assert mock_service.compact_called is True


class TestAICodingAppMessageRendering:
    """消息渲染方法测试."""

    async def test_add_user_message(self, mock_service: _MockAgentService) -> None:
        """用户输入应渲染为用户消息."""
        app = AICodingApp(mock_service)
        async with app.run_test() as pilot:
            app._add_user_message("hello")
            await pilot.pause()

            history = app.query_one("#history", VerticalScroll)
            user_msgs = list(history.query(".user-message"))
            assert len(user_msgs) == 1
            assert "hello" in _static_text(user_msgs[0])

    async def test_add_system_message(self, mock_service: _MockAgentService) -> None:
        """系统消息应正确渲染."""
        app = AICodingApp(mock_service)
        async with app.run_test() as pilot:
            app._add_system_message("system info")
            await pilot.pause()

            history = app.query_one("#history", VerticalScroll)
            sys_msgs = list(history.query(".system-message"))
            assert len(sys_msgs) == 1
            assert "system info" in _static_text(sys_msgs[0])

    async def test_add_error_message(self, mock_service: _MockAgentService) -> None:
        """错误消息应正确渲染."""
        app = AICodingApp(mock_service)
        async with app.run_test() as pilot:
            app._add_error_message("error occurred")
            await pilot.pause()

            history = app.query_one("#history", VerticalScroll)
            err_msgs = list(history.query(".error-message"))
            assert len(err_msgs) == 1
            assert "error occurred" in _static_text(err_msgs[0])

    async def test_assistant_streaming(self, mock_service: _MockAgentService) -> None:
        """Assistant 流式消息应能追加和完成."""
        app = AICodingApp(mock_service)
        async with app.run_test() as pilot:
            app._start_new_assistant_response()
            app._append_assistant_chunk("Hello")
            app._append_assistant_chunk(" World")
            app._finish_assistant_response()
            await pilot.pause()

            history = app.query_one("#history", VerticalScroll)
            msgs = list(history.query(".assistant-message"))
            assert len(msgs) == 1

    async def test_tool_call_and_observation(
        self, mock_service: _MockAgentService
    ) -> None:
        """工具调用面板和观察结果应正确渲染."""
        app = AICodingApp(mock_service)
        async with app.run_test() as pilot:
            app._add_tool_call("read_file", {"path": "README.md"})
            app._add_observation("file content")
            await pilot.pause()

            history = app.query_one("#history", VerticalScroll)
            collapsibles = list(history.query(Collapsible))
            assert len(collapsibles) == 1


class TestAICodingAppAgentEvents:
    """Agent 事件流处理测试."""

    async def test_stream_events_render(self, mock_service: _MockAgentService) -> None:
        """send_message_stream 产生的事件应正确渲染到界面."""
        mock_service.set_events(
            [
                ThinkingStartEvent(),
                ThinkingChunkEvent(text="thinking..."),
                ThinkingEndEvent(),
                AssistantStartEvent(),
                AssistantChunkEvent(text="Hello"),
                AssistantChunkEvent(text=" World"),
                AssistantEndEvent(),
                ToolCallEvent(name="read_file", args={"path": "README.md"}),
                ObservationEvent(text="file content"),
            ]
        )

        app = AICodingApp(mock_service)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#input", Input)
            input_widget.value = "say hello"
            await pilot.press("enter")
            await pilot.pause(0.5)

            assert mock_service.last_input == "say hello"
            history = app.query_one("#history", VerticalScroll)
            assert len(list(history.query(".user-message"))) == 1

    async def test_stream_error_render(self, mock_service: _MockAgentService) -> None:
        """流式错误事件应渲染为错误消息."""
        mock_service.set_events([ErrorEvent(text="stream failed")])

        app = AICodingApp(mock_service)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#input", Input)
            input_widget.value = "test"
            await pilot.press("enter")
            await pilot.pause(0.5)

            history = app.query_one("#history", VerticalScroll)
            err_msgs = list(history.query(".error-message"))
            assert len(err_msgs) >= 1
