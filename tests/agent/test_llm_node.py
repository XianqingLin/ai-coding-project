"""LLM 节点单元测试.

覆盖文件快照/任务列表上下文注入、子 Agent 结果通知、消息插入顺序等.
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai_coding.agent.nodes.llm_node import (
    FILE_SNAPSHOT_MAX_LINES,
    _build_file_context_message,
    _build_sub_agent_human_message,
    _build_todo_context_message,
    create_llm_node,
)
from ai_coding.agent.state import AgentState


def _make_state(
    messages: List[Any],
    file_snapshots: Optional[Dict[str, str]] = None,
    todos: Optional[List[dict]] = None,
    sub_agents: Optional[List[dict]] = None,
) -> AgentState:
    return {
        "messages": messages,
        "file_snapshots": file_snapshots or {},
        "todos": todos or [],
        "globally_approved_tools": [],
        "background_tasks": [],
        "plan_mode": False,
        "plan_file_path": "",
        "sub_agents": sub_agents or [],
    }


class TestBuildFileContextMessage:
    """文件快照上下文构建测试."""

    def test_empty_snapshots_returns_empty_message(self):
        msg = _build_file_context_message({})
        assert msg.content == ""
        assert isinstance(msg, SystemMessage)

    def test_normal_snapshots_formatted(self):
        msg = _build_file_context_message({"a.py": "x = 1\ny = 2"})
        assert "当前文件快照" in msg.content
        assert "--- a.py ---" in msg.content
        assert "x = 1" in msg.content

    def test_large_snapshot_truncated(self):
        lines = [f"line {i}" for i in range(FILE_SNAPSHOT_MAX_LINES + 10)]
        content = "\n".join(lines)
        msg = _build_file_context_message({"big.py": content})

        assert "省略" in msg.content
        # 首尾都应保留
        assert "line 0" in msg.content
        assert f"line {len(lines) - 1}" in msg.content


class TestBuildTodoContextMessage:
    """任务列表上下文构建测试."""

    def test_empty_todos_returns_empty_message(self):
        msg = _build_todo_context_message([])
        assert msg.content == ""

    def test_todos_formatted_with_progress(self):
        todos = [
            {"task": "task A", "done": True},
            {"task": "task B", "done": False},
        ]
        msg = _build_todo_context_message(todos)
        assert "当前任务列表" in msg.content
        assert "[x] 1. task A" in msg.content
        assert "[ ] 2. task B" in msg.content
        assert "进度: 1/2 已完成" in msg.content


class TestBuildSubAgentHumanMessage:
    """子 Agent 结果 HumanMessage 构建测试."""

    def test_no_pending_returns_empty(self, clean_sub_agent_manager):
        manager = clean_sub_agent_manager
        msg = _build_sub_agent_human_message(manager)
        assert msg.content == ""

    def test_pending_results_formatted(
        self, clean_sub_agent_manager, isolated_work_dir
    ):
        manager = clean_sub_agent_manager
        from ai_coding.mock_llm import MockChatModel, mock_text

        llm = MockChatModel(responses=[mock_text("explore result")])
        manager.dispatch(
            agent_type="explore",
            prompt="explore",
            llm=llm,
            work_dir=str(isolated_work_dir),
        )

        msg = _build_sub_agent_human_message(manager)
        assert "子 Agent 完成" in msg.content
        assert "explore result" in msg.content

    def test_failed_sub_agent_shows_failure(
        self, clean_sub_agent_manager, isolated_work_dir
    ):
        from ai_coding.mock_llm import MockChatModel

        manager = clean_sub_agent_manager

        class FailingLLM(MockChatModel):
            def invoke(self, messages, **kwargs):
                raise RuntimeError("fail")

        manager.dispatch(
            agent_type="explore",
            prompt="explore",
            llm=FailingLLM(),
            work_dir=str(isolated_work_dir),
        )

        msg = _build_sub_agent_human_message(manager)
        assert "失败" in msg.content

    def test_long_result_truncated(self, clean_sub_agent_manager, isolated_work_dir):
        manager = clean_sub_agent_manager
        from ai_coding.mock_llm import MockChatModel, mock_text

        long_result = "x" * 2500
        llm = MockChatModel(responses=[mock_text(long_result)])
        manager.dispatch(
            agent_type="explore",
            prompt="explore",
            llm=llm,
            work_dir=str(isolated_work_dir),
        )

        msg = _build_sub_agent_human_message(manager)
        assert "结果已截断" in msg.content


class TestCreateLLMNode:
    """LLM 节点整体调用测试."""

    def test_invokes_llm_and_returns_response(self):
        """节点调用 LLM 并返回 AIMessage."""
        llm = MagicMock()
        response = AIMessage(content="hello", tool_calls=[])
        llm.invoke.return_value = response

        node = create_llm_node(llm)
        state = _make_state(messages=[HumanMessage(content="hi")])
        result = node(state)

        assert result["messages"] == [response]
        llm.invoke.assert_called_once()

    def test_injects_file_context_after_system_message(self):
        """文件快照上下文应插入到 SystemMessage 之后."""
        llm = MagicMock()
        response = AIMessage(content="ok")
        llm.invoke.return_value = response

        node = create_llm_node(llm)
        state = _make_state(
            messages=[
                SystemMessage(content="sys"),
                HumanMessage(content="hi"),
            ],
            file_snapshots={"a.py": "x = 1"},
        )
        node(state)

        call_messages = llm.invoke.call_args[0][0]
        assert isinstance(call_messages[0], SystemMessage)
        assert isinstance(call_messages[1], SystemMessage)
        assert "a.py" in call_messages[1].content
        assert isinstance(call_messages[2], HumanMessage)

    def test_injects_todo_context_after_file_context(self):
        """任务列表上下文应插入到文件上下文之后."""
        llm = MagicMock()
        response = AIMessage(content="ok")
        llm.invoke.return_value = response

        node = create_llm_node(llm)
        state = _make_state(
            messages=[HumanMessage(content="hi")],
            file_snapshots={"a.py": "x = 1"},
            todos=[{"task": "t", "done": False}],
        )
        node(state)

        call_messages = llm.invoke.call_args[0][0]
        # 没有 SystemMessage 时，上下文插入到索引 0
        assert isinstance(call_messages[0], SystemMessage)
        assert "a.py" in call_messages[0].content
        assert isinstance(call_messages[1], SystemMessage)
        assert "当前任务列表" in call_messages[1].content
        assert isinstance(call_messages[2], HumanMessage)

    def test_no_injection_when_empty(self):
        """空快照/空任务列表时不插入额外消息."""
        llm = MagicMock()
        response = AIMessage(content="ok")
        llm.invoke.return_value = response

        node = create_llm_node(llm)
        state = _make_state(messages=[HumanMessage(content="hi")])
        node(state)

        call_messages = llm.invoke.call_args[0][0]
        assert len(call_messages) == 1

    def test_sub_agent_notification_marks_notified(
        self, clean_sub_agent_manager, isolated_work_dir
    ):
        """子 Agent 完成后，LLM 节点应注入结果并标记已通知."""
        from ai_coding.mock_llm import MockChatModel, mock_text

        manager = clean_sub_agent_manager
        llm = MagicMock()
        response = AIMessage(content="ack")
        llm.invoke.return_value = response

        sub_llm = MockChatModel(responses=[mock_text("result from sub")])
        manager.dispatch(
            agent_type="explore",
            prompt="explore",
            llm=sub_llm,
            work_dir=str(isolated_work_dir),
        )
        assert len(manager.get_pending_notifications()) == 1

        node = create_llm_node(llm)
        state = _make_state(messages=[HumanMessage(content="hi")])
        result = node(state)

        call_messages = llm.invoke.call_args[0][0]
        assert any("子 Agent 完成" in m.content for m in call_messages)

        # 标记已通知后，pending 应清空；state 中的 sub_agents 被更新
        assert len(manager.get_pending_notifications()) == 0
        assert len(result["sub_agents"]) == 1
        assert result["sub_agents"][0]["notified"] is True

    def test_response_with_tool_calls_logged(self):
        """带 tool_calls 的响应会记录工具名."""
        llm = MagicMock()
        response = AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "x"}, "id": "t1"}],
            additional_kwargs={"reasoning_content": "think"},
        )
        llm.invoke.return_value = response

        node = create_llm_node(llm)
        state = _make_state(messages=[HumanMessage(content="hi")])
        result = node(state)

        assert result["messages"] == [response]
