"""KimiChatOpenAI 单元测试.

覆盖 reasoning_content 在请求注入、非流式响应提取、流式 chunk 提取三个环节.
"""

from typing import List, Optional
from unittest.mock import patch

import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ai_coding.llm.kimi_chat import KimiChatOpenAI


class FakeReasoningMessage(openai.BaseModel):
    """模拟 OpenAI 响应 message，包含 reasoning_content."""

    content: str = ""
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[dict]] = None


class FakeChoice(openai.BaseModel):
    message: FakeReasoningMessage


class FakeChatCompletion(openai.BaseModel):
    choices: List[FakeChoice]


@pytest.fixture
def kimi():
    """创建 KimiChatOpenAI 实例（使用假 key，不发起真实请求）."""
    return KimiChatOpenAI(api_key="fake-key", model="kimi-k2-6")


class TestGetRequestPayload:
    """_get_request_payload 注入 reasoning_content 测试."""

    def test_injects_reasoning_for_assistant_with_tool_calls(self, kimi):
        payload = kimi._get_request_payload(
            [
                HumanMessage(content="hi"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "tc1", "name": "read_file", "args": {"path": "x"}}
                    ],
                ),
                ToolMessage(content="done", tool_call_id="tc1"),
            ]
        )

        messages = payload.get("messages", [])
        assistant_msg = [m for m in messages if m.get("role") == "assistant"][0]
        assert assistant_msg.get("reasoning_content") == ""

    def test_no_injection_for_assistant_without_tool_calls(self, kimi):
        payload = kimi._get_request_payload(
            [
                HumanMessage(content="hi"),
                AIMessage(content="hello"),
            ]
        )

        messages = payload.get("messages", [])
        assistant_msg = [m for m in messages if m.get("role") == "assistant"][0]
        assert "reasoning_content" not in assistant_msg

    def test_no_injection_if_reasoning_already_present(self, kimi):
        """payload message 中已有 reasoning_content 时不覆盖."""
        # 直接构造一个已包含 reasoning_content 的 payload 返回给父类
        parent_payload = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "tc1"}],
                    "reasoning_content": "existing reasoning",
                }
            ]
        }

        with patch.object(
            type(kimi).__bases__[0], "_get_request_payload", return_value=parent_payload
        ):
            payload = kimi._get_request_payload([])

        assistant_msg = payload["messages"][0]
        assert assistant_msg.get("reasoning_content") == "existing reasoning"


class TestCreateChatResult:
    """非流式响应 reasoning_content 提取测试."""

    def test_extracts_reasoning_content(self, kimi):
        """从 openai BaseModel 响应中提取 reasoning_content 注入 ChatResult."""
        response = FakeChatCompletion(
            choices=[
                FakeChoice(
                    message=FakeReasoningMessage(
                        reasoning_content="step by step thinking"
                    )
                )
            ]
        )

        # 构造父类返回的 ChatResult
        gen_msg = AIMessage(content="ok")
        chat_result = ChatResult(generations=[ChatGeneration(message=gen_msg)])

        with patch.object(
            type(kimi).__bases__[0], "_create_chat_result", return_value=chat_result
        ):
            result = kimi._create_chat_result(response)

        assert (
            result.generations[0].message.additional_kwargs.get("reasoning_content")
            == "step by step thinking"
        )

    def test_no_reasoning_content(self, kimi):
        """响应中没有 reasoning_content 时不添加."""
        response = FakeChatCompletion(
            choices=[FakeChoice(message=FakeReasoningMessage(reasoning_content=None))]
        )

        gen_msg = AIMessage(content="ok")
        chat_result = ChatResult(generations=[ChatGeneration(message=gen_msg)])

        with patch.object(
            type(kimi).__bases__[0], "_create_chat_result", return_value=chat_result
        ):
            result = kimi._create_chat_result(response)

        assert (
            "reasoning_content" not in result.generations[0].message.additional_kwargs
        )

    def test_response_without_choices(self, kimi):
        """choices 为空时不报错."""
        response = FakeChatCompletion(choices=[])
        chat_result = ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="ok"))]
        )

        with patch.object(
            type(kimi).__bases__[0], "_create_chat_result", return_value=chat_result
        ):
            result = kimi._create_chat_result(response)

        assert result is chat_result


class TestConvertChunkToGenerationChunk:
    """流式 chunk reasoning_content 提取测试."""

    def test_extracts_reasoning_from_chunk(self, kimi):
        """流式 chunk 中包含 reasoning_content 时写入 AIMessageChunk."""
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        chunk_msg = AIMessageChunk(content="ok")
        parent_chunk = ChatGenerationChunk(message=chunk_msg)

        raw_chunk = {
            "choices": [
                {
                    "delta": {
                        "content": "ok",
                        "reasoning_content": "stream reasoning",
                    }
                }
            ]
        }

        with patch.object(
            type(kimi).__bases__[0],
            "_convert_chunk_to_generation_chunk",
            return_value=parent_chunk,
        ):
            result = kimi._convert_chunk_to_generation_chunk(
                raw_chunk, ChatGenerationChunk, None
            )

        assert (
            result.message.additional_kwargs.get("reasoning_content")
            == "stream reasoning"
        )

    def test_returns_none_when_parent_returns_none(self, kimi):
        """父类返回 None 时直接返回 None."""
        from langchain_core.outputs import ChatGenerationChunk

        raw_chunk = {
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "ignored",
                    }
                }
            ]
        }

        with patch.object(
            type(kimi).__bases__[0],
            "_convert_chunk_to_generation_chunk",
            return_value=None,
        ):
            result = kimi._convert_chunk_to_generation_chunk(
                raw_chunk, ChatGenerationChunk, None
            )

        assert result is None

    def test_no_reasoning_content_in_chunk(self, kimi):
        """chunk 中无 reasoning_content 时不修改 additional_kwargs."""
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        chunk_msg = AIMessageChunk(content="ok")
        parent_chunk = ChatGenerationChunk(message=chunk_msg)

        raw_chunk = {"choices": [{"delta": {"content": "ok"}}]}

        with patch.object(
            type(kimi).__bases__[0],
            "_convert_chunk_to_generation_chunk",
            return_value=parent_chunk,
        ):
            result = kimi._convert_chunk_to_generation_chunk(
                raw_chunk, ChatGenerationChunk, None
            )

        assert "reasoning_content" not in result.message.additional_kwargs

    def test_extracts_from_nested_chunk(self, kimi):
        """兼容 chunk 嵌套在 'chunk' 键下的情况."""
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        chunk_msg = AIMessageChunk(content="ok")
        parent_chunk = ChatGenerationChunk(message=chunk_msg)

        raw_chunk = {
            "chunk": {"choices": [{"delta": {"reasoning_content": "nested reasoning"}}]}
        }

        with patch.object(
            type(kimi).__bases__[0],
            "_convert_chunk_to_generation_chunk",
            return_value=parent_chunk,
        ):
            result = kimi._convert_chunk_to_generation_chunk(
                raw_chunk, ChatGenerationChunk, None
            )

        assert (
            result.message.additional_kwargs.get("reasoning_content")
            == "nested reasoning"
        )
