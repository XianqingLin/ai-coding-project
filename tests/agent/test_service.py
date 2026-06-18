"""AgentService 单元测试."""

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from ai_coding.agent.events import (
    AssistantChunkEvent,
    AssistantEndEvent,
    AssistantStartEvent,
    ErrorEvent,
    ObservationEvent,
    ThinkingChunkEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallEvent,
    UserInputEvent,
)
from ai_coding.agent.service import AgentService


@pytest.fixture
def service(isolated_work_dir):
    """创建使用 mock provider 的 AgentService."""
    return AgentService(
        work_dir=str(isolated_work_dir),
        llm_provider="mock",
        auto_approve=True,
    )


class TestAgentServiceSessionManagement:
    """会话管理接口测试."""

    def test_create_session(self, service):
        sid = service.create_session("test")
        assert sid
        assert service.current_session_id == sid

    def test_list_sessions(self, service):
        sid = service.create_session("test")
        sessions = service.list_sessions()
        assert any(s.get("session_id") == sid for s in sessions)

    def test_switch_session(self, service):
        sid1 = service.create_session("first")
        sid2 = service.create_session("second")
        assert service.current_session_id == sid2

        assert service.switch_session(sid1) is True
        assert service.current_session_id == sid1

    def test_switch_unknown_session(self, service):
        assert service.switch_session("not-exist") is False

    def test_delete_session(self, service):
        sid = service.create_session("to-delete")
        assert service.delete_session(sid) is True
        assert service.current_session_id != sid

    def test_delete_unknown_session(self, service):
        assert service.delete_session("not-exist") is False

    def test_rename_session(self, service):
        sid = service.create_session("old")
        assert service.rename_session(sid, "new") is True
        sessions = service.list_sessions()
        session = next(s for s in sessions if s["session_id"] == sid)
        assert session["name"] == "new"

    def test_rename_unknown_session(self, service):
        assert service.rename_session("not-exist", "x") is False

    def test_get_current_session(self, service):
        sid = service.create_session("current")
        current = service.get_current_session()
        assert current["session_id"] == sid


class TestAgentServiceMessaging:
    """消息发送接口测试."""

    def test_send_message(self, service):
        service.create_session("test")
        result = service.send_message("hi")
        assert isinstance(result, str)

    def test_send_message_returns_error_when_no_agent(self, service, monkeypatch):
        monkeypatch.setattr(service, "_get_agent", lambda _=None: None)
        result = service.send_message("hi")
        assert "没有可用的 Agent 会话" in result

    def test_send_message_stream(self, service):
        service.create_session("test")
        events = list(service.send_message_stream("hi"))
        assert isinstance(events[0], UserInputEvent)

    def test_send_message_stream_returns_error_when_no_agent(self, service, monkeypatch):
        monkeypatch.setattr(service, "_get_agent", lambda _=None: None)
        events = list(service.send_message_stream("hi"))
        assert isinstance(events[0], UserInputEvent)
        assert isinstance(events[1], ErrorEvent)
        assert "没有可用的 Agent 会话" in events[1].text

    def test_send_message_stream_not_supported(self, service, monkeypatch):
        # 模拟 Agent 不支持 run_stream_verbose
        mock_agent = MagicMock()
        del mock_agent.run_stream_verbose
        monkeypatch.setattr(service, "_get_agent", lambda _=None: mock_agent)
        events = list(service.send_message_stream("hi"))
        assert any(
            "不支持流式事件模式" in e.text
            for e in events
            if isinstance(e, ErrorEvent)
        )

    def test_send_message_stream_error(self, service):
        service.create_session("test")
        agent = service._get_agent()

        def failing_stream(text):
            raise RuntimeError("stream error")

        agent.run_stream_verbose = failing_stream
        events = list(service.send_message_stream("hi"))
        assert any(
            "流式运行失败" in e.text for e in events if isinstance(e, ErrorEvent)
        )


class TestAgentServiceEventMapping:
    """事件映射测试."""

    def test_map_all_event_types(self, service):
        raw_events = [
            {"type": "user_input", "input": "hi"},
            {"type": "thinking_start"},
            {"type": "thinking_chunk", "text": "thinking"},
            {"type": "thinking_end"},
            {"type": "assistant_start"},
            {"type": "assistant_chunk", "text": "chunk"},
            {"type": "assistant_end", "text": "end"},
            {"type": "tool_call", "name": "read_file", "args": {"path": "x"}},
            {"type": "observation", "text": "obs"},
            {"type": "error", "text": "err"},
            {"type": "unknown_type"},
        ]

        results = [service._map_raw_event(raw) for raw in raw_events]

        assert isinstance(results[0], UserInputEvent)
        assert isinstance(results[1], ThinkingStartEvent)
        assert isinstance(results[2], ThinkingChunkEvent)
        assert isinstance(results[3], ThinkingEndEvent)
        assert isinstance(results[4], AssistantStartEvent)
        assert isinstance(results[5], AssistantChunkEvent)
        assert isinstance(results[6], AssistantEndEvent)
        assert isinstance(results[7], ToolCallEvent)
        assert isinstance(results[8], ObservationEvent)
        assert isinstance(results[9], ErrorEvent)
        assert results[10] is None


class TestAgentServiceReadOnlyQueries:
    """只读查询接口测试."""

    def test_get_history(self, service):
        service.create_session("test")
        history = service.get_history()
        assert isinstance(history, list)

    def test_get_context_usage(self, service):
        service.create_session("test")
        usage = service.get_context_usage()
        assert "used_tokens" in usage
        assert "limit_tokens" in usage
        assert "percentage" in usage

    def test_get_stats(self, service):
        service.create_session("test")
        stats = service.get_stats()
        assert isinstance(stats, dict)

    def test_get_system_prompt(self, service):
        service.create_session("test")
        prompt = service.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_agent_safe_does_not_switch(self, service):
        sid1 = service.create_session("first")
        sid2 = service.create_session("second")
        assert service.current_session_id == sid2

        agent = service._get_agent_safe(sid1)
        # 安全获取不应切换当前会话
        assert service.current_session_id == sid2
        assert agent is not None

    def test_get_agent_switches(self, service):
        sid1 = service.create_session("first")
        sid2 = service.create_session("second")
        assert service.current_session_id == sid2

        agent = service._get_agent(sid1)
        assert service.current_session_id == sid1
        assert agent is not None

    def test_resolve_session_id_defaults_to_current(self, service):
        sid = service.create_session("test")
        assert service._resolve_session_id(None) == sid
        assert service._resolve_session_id(sid) == sid
