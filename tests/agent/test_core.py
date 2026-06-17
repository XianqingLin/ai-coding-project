"""LangGraphAgent 核心单元测试."""

import pytest
from langchain_core.messages import AIMessage

from ai_coding.agent.core import LangGraphAgent
from ai_coding.mock_llm import MockChatModel, mock_text, mock_tool_call
from ai_coding.tools import create_default_tools


class TestLangGraphAgent:
    def test_run_text_response(self):
        llm = MockChatModel(responses=[mock_text("Hello!")])
        agent = LangGraphAgent(llm=llm, tools=[])

        result = agent.run("hi")

        assert result == "Hello!"
        assert agent.state is not None

    def test_run_react_loop_with_tool(self):
        """测试完整的 ReAct 循环：用户输入 -> AI 调用 list_dir -> AI 最终回复."""
        llm = MockChatModel(responses=[
            mock_tool_call("list_dir", {"path": "."}, content="看看目录", call_id="tc1"),
            mock_text("我已完成目录查看。"),
        ])
        # 只给 list_dir 工具，避免 approval 阻塞
        tools = [t for t in create_default_tools() if t.name in ("list_dir",)]
        agent = LangGraphAgent(llm=llm, tools=tools, auto_approve=True)

        result = agent.run("查看当前目录")

        assert "我已完成目录查看。" in result
        # 状态应包含 ToolMessage
        messages = agent.state["messages"]
        assert any(isinstance(m, AIMessage) and m.tool_calls for m in messages)

    def test_run_stream_text(self):
        llm = MockChatModel(responses=[mock_text("streamed")])
        agent = LangGraphAgent(llm=llm, tools=[])

        chunks = list(agent.run_stream("hi"))

        assert "".join(chunks) == "streamed"

    def test_clear_history(self):
        llm = MockChatModel(responses=[mock_text("reply")])
        agent = LangGraphAgent(llm=llm, tools=[])
        agent.run("hi")

        old_thread_id = agent.thread_id
        agent.clear_history()

        assert agent.state is None
        assert agent.thread_id != old_thread_id

    def test_get_history(self):
        llm = MockChatModel(responses=[mock_text("reply")])
        agent = LangGraphAgent(llm=llm, tools=[])
        agent.run("hi")

        history = agent.get_history()
        assert any(h.get("role") == "user" and h.get("content") == "hi" for h in history)
        assert any(h.get("role") == "assistant" for h in history)
