"""AgentService 单元测试."""

import pytest

from ai_coding.agent import AgentService
from ai_coding.agent.events import (
    AssistantChunkEvent,
    AssistantEndEvent,
    AssistantStartEvent,
    UserInputEvent,
)
from ai_coding.mock_llm import MockChatModel, mock_text, mock_tool_call


class TestAgentService:
    def test_create_and_list_sessions(self, isolated_work_dir):
        llm = MockChatModel(responses=[mock_text("ok")])
        svc = AgentService(work_dir=str(isolated_work_dir), llm_factory=lambda: llm)

        sid = svc.create_session("test-session")

        assert sid
        sessions = svc.list_sessions()
        assert any(s["session_id"] == sid and s["name"] == "test-session" for s in sessions)
        assert svc.current_session_id == sid

    def test_switch_session(self, isolated_work_dir):
        llm = MockChatModel(responses=[mock_text("ok")])
        svc = AgentService(work_dir=str(isolated_work_dir), llm_factory=lambda: llm)

        sid1 = svc.create_session("session-1")
        sid2 = svc.create_session("session-2")

        assert svc.current_session_id == sid2
        assert svc.switch_session(sid1) is True
        assert svc.current_session_id == sid1
        assert svc.switch_session("not-exist") is False

    def test_rename_and_delete_session(self, isolated_work_dir):
        llm = MockChatModel(responses=[mock_text("ok")])
        svc = AgentService(work_dir=str(isolated_work_dir), llm_factory=lambda: llm)

        sid = svc.create_session("old-name")
        assert svc.rename_session(sid, "new-name") is True

        sessions = svc.list_sessions()
        assert any(s["session_id"] == sid and s["name"] == "new-name" for s in sessions)

        assert svc.delete_session(sid) is True
        sessions = svc.list_sessions()
        assert not any(s["session_id"] == sid for s in sessions)

    def test_send_message(self, isolated_work_dir):
        llm = MockChatModel(responses=[mock_text("Hello!")])
        svc = AgentService(work_dir=str(isolated_work_dir), llm_factory=lambda: llm)
        svc.create_session("test")

        reply = svc.send_message("hi")

        assert reply == "Hello!"
        history = svc.get_history()
        assert any(h.get("role") == "user" and h.get("content") == "hi" for h in history)
        assert any(h.get("role") == "assistant" and h.get("content") == "Hello!" for h in history)

    def test_send_message_stream(self, isolated_work_dir):
        llm = MockChatModel(responses=[mock_text("streamed")])
        svc = AgentService(work_dir=str(isolated_work_dir), llm_factory=lambda: llm)
        svc.create_session("test")

        events = list(svc.send_message_stream("hi"))

        # 至少应包含 user_input、assistant_start、assistant_chunk、assistant_end
        assert events[0].type == "user_input"
        assert events[0].text == "hi"
        assert any(e.type == "assistant_start" for e in events)
        assert any(e.type == "assistant_chunk" and e.text == "streamed" for e in events)
        assert any(e.type == "assistant_end" for e in events)

    def test_send_message_with_tool_call(self, isolated_work_dir):
        """测试完整的 ReAct 循环通过 AgentService 正常工作."""
        llm = MockChatModel(responses=[
            mock_tool_call("list_dir", {"path": "."}, content="查看目录", call_id="tc1"),
            mock_text("目录查看完成。"),
        ])
        svc = AgentService(work_dir=str(isolated_work_dir), llm_factory=lambda: llm, auto_approve=True)
        svc.create_session("test")

        reply = svc.send_message("查看当前目录")

        assert "目录查看完成。" in reply
        history = svc.get_history()
        assert any(h.get("role") == "tool" for h in history)

    def test_context_usage(self, isolated_work_dir):
        llm = MockChatModel(responses=[mock_text("ok")])
        svc = AgentService(work_dir=str(isolated_work_dir), llm_factory=lambda: llm)
        svc.create_session("test")
        svc.send_message("hi")

        usage = svc.get_context_usage()
        assert "used_tokens" in usage
        assert "limit_tokens" in usage
        assert "percentage" in usage

    def test_get_history_empty_session(self, isolated_work_dir):
        llm = MockChatModel(responses=[mock_text("ok")])
        svc = AgentService(work_dir=str(isolated_work_dir), llm_factory=lambda: llm)
        svc.create_session("test")

        history = svc.get_history()
        assert history == []
