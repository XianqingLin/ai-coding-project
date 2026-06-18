"""Mock LLM 单元测试.

覆盖预设响应序列、交互式模式、流式输出、reasoning_content 兼容等.
"""

from langchain_core.messages import AIMessage, HumanMessage

from ai_coding.mock_llm import MockChatModel, mock_text, mock_tool_call


class TestMockHelpers:
    """mock_tool_call / mock_text 辅助函数测试."""

    def test_mock_tool_call_creates_aimessage_with_tool_calls(self):
        msg = mock_tool_call("read_file", {"path": "x.py"}, content="让我看看")

        assert isinstance(msg, AIMessage)
        assert msg.content == "让我看看"
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "read_file"
        assert msg.tool_calls[0]["args"] == {"path": "x.py"}
        assert "reasoning_content" in msg.additional_kwargs

    def test_mock_tool_call_custom_call_id(self):
        msg = mock_tool_call("write_file", {}, call_id="custom_id")
        assert msg.tool_calls[0]["id"] == "custom_id"

    def test_mock_text_creates_plain_text_message(self):
        msg = mock_text("hello")
        assert isinstance(msg, AIMessage)
        assert msg.content == "hello"
        assert not msg.tool_calls


class TestMockChatModelPreset:
    """MockChatModel 预设响应测试."""

    def test_bind_tools_returns_self(self):
        llm = MockChatModel()
        bound = llm.bind_tools(["tool1"])
        assert bound is llm
        assert llm._tools == ["tool1"]

    def test_invoke_returns_preset_responses_in_order(self):
        llm = MockChatModel(
            responses=[
                mock_tool_call("list_dir", {"path": "."}, content="看看目录"),
                mock_text("完成了"),
            ]
        )

        r1 = llm.invoke([HumanMessage(content="hi")])
        r2 = llm.invoke([HumanMessage(content="hi")])

        assert r1.tool_calls[0]["name"] == "list_dir"
        assert r2.content == "完成了"

    def test_string_response_converted_to_aimessage(self):
        llm = MockChatModel(responses=["plain text"])
        result = llm.invoke([HumanMessage(content="hi")])
        assert result.content == "plain text"

    def test_reasoning_content_injected_for_tool_calls(self):
        msg = mock_tool_call("read_file", {"path": "x.py"})
        msg.additional_kwargs.pop("reasoning_content", None)
        llm = MockChatModel(responses=[msg])

        result = llm.invoke([HumanMessage(content="hi")])
        assert result.additional_kwargs.get("reasoning_content") == ""

    def test_default_message_when_responses_exhausted(self):
        llm = MockChatModel(responses=[mock_text("only one")])
        llm.invoke([HumanMessage(content="hi")])
        result = llm.invoke([HumanMessage(content="hi")])

        assert "预设回复序列已耗尽" in result.content


class TestMockChatModelInteractive:
    """MockChatModel 交互式模式测试."""

    def test_interactive_text_response(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "text:hello world")
        llm = MockChatModel(interactive=True)

        result = llm.invoke([HumanMessage(content="hi")])
        assert result.content == "hello world"

    def test_interactive_tool_response(self, monkeypatch):
        monkeypatch.setattr(
            "builtins.input", lambda _: "tool:read_file {\"path\": \"x.py\"}"
        )
        llm = MockChatModel(interactive=True)

        result = llm.invoke([HumanMessage(content="hi")])
        assert result.tool_calls[0]["name"] == "read_file"
        assert result.tool_calls[0]["args"] == {"path": "x.py"}

    def test_interactive_tool_response_no_args(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "tool:list_dir")
        llm = MockChatModel(interactive=True)

        result = llm.invoke([HumanMessage(content="hi")])
        assert result.tool_calls[0]["name"] == "list_dir"
        assert result.tool_calls[0]["args"] == {}

    def test_interactive_tool_invalid_json_falls_back_to_empty_args(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "tool:read_file not-json")
        llm = MockChatModel(interactive=True)

        result = llm.invoke([HumanMessage(content="hi")])
        assert result.tool_calls[0]["name"] == "read_file"
        assert result.tool_calls[0]["args"] == {}

    def test_interactive_plain_text_fallback(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "直接输入")
        llm = MockChatModel(interactive=True)

        result = llm.invoke([HumanMessage(content="hi")])
        assert result.content == "直接输入"


class TestMockChatModelStream:
    """MockChatModel 流式输出测试."""

    def test_stream_yields_content(self):
        llm = MockChatModel(responses=[mock_text("streamed content")])
        chunks = list(llm.stream([HumanMessage(content="hi")]))

        assert "".join(chunks) == "streamed content"

    def test_stream_empty_content_yields_nothing(self):
        llm = MockChatModel(responses=[mock_tool_call("read_file", {"path": "x.py"})])
        chunks = list(llm.stream([HumanMessage(content="hi")]))

        assert chunks == []
