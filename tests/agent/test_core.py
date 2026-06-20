"""LangGraphAgent 核心单元测试."""

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
        llm = MockChatModel(
            responses=[
                mock_tool_call(
                    "list_dir", {"path": "."}, content="看看目录", call_id="tc1"
                ),
                mock_text("我已完成目录查看。"),
            ]
        )
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

    def test_run_stream_verbose_text(self):
        """run_stream_verbose 输出 thinking/assistant 事件."""
        llm = MockChatModel(responses=[mock_text("verbose result")])
        agent = LangGraphAgent(llm=llm, tools=[])

        events = list(agent.run_stream_verbose("hi"))

        assert any(e.get("type") == "assistant_start" for e in events)
        assert any(
            e.get("type") == "assistant_chunk" and "verbose result" in e.get("text", "")
            for e in events
        )
        assert any(e.get("type") == "assistant_end" for e in events)

    def test_run_stream_verbose_with_tool_call(self):
        """run_stream_verbose 输出 tool_call 和 observation 事件."""
        llm = MockChatModel(
            responses=[
                mock_tool_call(
                    "list_dir", {"path": "."}, content="调用工具", call_id="tc1"
                ),
                mock_text("done"),
            ]
        )
        tools = [t for t in create_default_tools() if t.name in ("list_dir",)]
        agent = LangGraphAgent(llm=llm, tools=tools, auto_approve=True)

        events = list(agent.run_stream_verbose("查看目录"))

        assert any(
            e.get("type") == "tool_call" and e.get("name") == "list_dir" for e in events
        )
        assert any(e.get("type") == "observation" for e in events)

    def test_run_stream_verbose_error(self):
        """run_stream_verbose 在异常时输出 error 事件."""

        class BadLLM(MockChatModel):
            def invoke(self, messages, **kwargs):
                raise RuntimeError("boom")

            def stream(self, messages, **kwargs):
                raise RuntimeError("boom")

        agent = LangGraphAgent(llm=BadLLM(), tools=[])
        events = list(agent.run_stream_verbose("hi"))

        assert any(
            "boom" in e.get("text", "") for e in events if e.get("type") == "error"
        )

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
        assert any(
            h.get("role") == "user" and h.get("content") == "hi" for h in history
        )
        assert any(h.get("role") == "assistant" for h in history)

    def test_get_context_usage(self):
        llm = MockChatModel(responses=[mock_text("reply")])
        agent = LangGraphAgent(llm=llm, tools=[])
        agent.run("hi")

        usage = agent.get_context_usage()
        assert usage["used_tokens"] > 0
        assert usage["limit_tokens"] > 0
        assert usage["percentage"] >= 0.0

    def test_get_context_usage_before_run(self):
        agent = LangGraphAgent(llm=MockChatModel(), tools=[])
        usage = agent.get_context_usage()
        # 即使未运行，system_prompt 也会占用 token
        assert usage["used_tokens"] > 0
        assert usage["limit_tokens"] > 0

    def test_get_stats(self):
        llm = MockChatModel(responses=[mock_text("reply")])
        agent = LangGraphAgent(llm=llm, tools=[])
        agent.run("hi")

        stats = agent.get_stats()
        assert stats["message_count"] > 0
        assert stats["type"] == "langgraph"
        assert stats["thread_id"] == agent.thread_id
