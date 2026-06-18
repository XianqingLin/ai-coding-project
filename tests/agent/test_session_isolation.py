"""多会话隔离与持久化恢复测试.

重点验证：
1. 只读查询（get_history/get_context_usage/get_stats/get_system_prompt）
   不会隐式切换当前会话。
2. 不同会话的历史、上下文、system prompt 互不混淆。
3. 持久化后重启 AgentService，会话状态可正确恢复。
"""

from ai_coding.agent import AgentService
from ai_coding.mock_llm import MockChatModel, mock_text


class TestSessionIsolation:
    """多会话隔离测试."""

    def test_read_operations_do_not_switch_current_session(self, isolated_work_dir):
        """对非当前会话的只读查询不应改变 current_session_id."""
        llm = MockChatModel(responses=[mock_text("ok")])
        svc = AgentService(work_dir=str(isolated_work_dir), llm_factory=lambda: llm)

        sid1 = svc.create_session("session-1")
        sid2 = svc.create_session("session-2")
        assert svc.current_session_id == sid2

        svc.send_message("hello from session-2")
        svc.switch_session(sid1)
        svc.send_message("hello from session-1")

        # 当前是会话 1
        assert svc.current_session_id == sid1

        # 查询会话 2 的各种只读接口，current 不应被切走
        svc.get_history(sid2)
        assert svc.current_session_id == sid1

        svc.get_context_usage(sid2)
        assert svc.current_session_id == sid1

        svc.get_stats(sid2)
        assert svc.current_session_id == sid1

        svc.get_system_prompt(sid2)
        assert svc.current_session_id == sid1

    def test_session_history_isolated(self, isolated_work_dir):
        """不同会话的历史记录互不混淆."""
        llm = MockChatModel(responses=[mock_text("ok")])
        svc = AgentService(work_dir=str(isolated_work_dir), llm_factory=lambda: llm)

        sid1 = svc.create_session("session-1")
        svc.send_message("message-A")

        sid2 = svc.create_session("session-2")
        svc.send_message("message-B")

        history1 = svc.get_history(sid1)
        history2 = svc.get_history(sid2)

        assert any(h.get("content") == "message-A" for h in history1)
        assert not any(h.get("content") == "message-B" for h in history1)

        assert any(h.get("content") == "message-B" for h in history2)
        assert not any(h.get("content") == "message-A" for h in history2)

    def test_switch_and_send_message(self, isolated_work_dir):
        """切换会话后发送消息应进入正确会话."""
        llm = MockChatModel(responses=[mock_text("ok")])
        svc = AgentService(work_dir=str(isolated_work_dir), llm_factory=lambda: llm)

        sid1 = svc.create_session("session-1")
        sid2 = svc.create_session("session-2")

        svc.switch_session(sid1)
        svc.send_message("to-session-1")

        svc.switch_session(sid2)
        svc.send_message("to-session-2")

        history1 = svc.get_history(sid1)
        history2 = svc.get_history(sid2)

        assert any(h.get("content") == "to-session-1" for h in history1)
        assert not any(h.get("content") == "to-session-2" for h in history1)

        assert any(h.get("content") == "to-session-2" for h in history2)
        assert not any(h.get("content") == "to-session-1" for h in history2)


class TestSessionPersistence:
    """会话持久化恢复测试."""

    def test_persistence_across_service_restart(self, isolated_work_dir):
        """重启 AgentService 后，会话历史和当前会话应正确恢复."""
        llm = MockChatModel(responses=[mock_text("ok")])
        svc1 = AgentService(work_dir=str(isolated_work_dir), llm_factory=lambda: llm)

        sid1 = svc1.create_session("session-1")
        sid2 = svc1.create_session("session-2")

        svc1.switch_session(sid1)
        svc1.send_message("persisted-A")

        svc1.switch_session(sid2)
        svc1.send_message("persisted-B")

        # 记住当前会话
        assert svc1.current_session_id == sid2

        # 销毁 service，模拟进程重启
        del svc1

        svc2 = AgentService(work_dir=str(isolated_work_dir), llm_factory=lambda: llm)

        # 会话列表恢复
        sessions = svc2.list_sessions()
        session_ids = {s["session_id"] for s in sessions}
        assert sid1 in session_ids
        assert sid2 in session_ids

        # 当前会话恢复
        assert svc2.current_session_id == sid2

        # 历史恢复且互不混淆
        history1 = svc2.get_history(sid1)
        history2 = svc2.get_history(sid2)

        assert any(h.get("content") == "persisted-A" for h in history1)
        assert not any(h.get("content") == "persisted-B" for h in history1)

        assert any(h.get("content") == "persisted-B" for h in history2)
        assert not any(h.get("content") == "persisted-A" for h in history2)

    def test_read_operations_after_restart_do_not_switch(self, isolated_work_dir):
        """重启后读取其他会话仍不应切换 current_session."""
        llm = MockChatModel(responses=[mock_text("ok")])
        svc1 = AgentService(work_dir=str(isolated_work_dir), llm_factory=lambda: llm)

        sid1 = svc1.create_session("session-1")
        sid2 = svc1.create_session("session-2")
        svc1.send_message("msg-2")

        del svc1

        svc2 = AgentService(work_dir=str(isolated_work_dir), llm_factory=lambda: llm)
        assert svc2.current_session_id == sid2

        svc2.get_history(sid1)
        assert svc2.current_session_id == sid2

        svc2.get_context_usage(sid1)
        assert svc2.current_session_id == sid2
