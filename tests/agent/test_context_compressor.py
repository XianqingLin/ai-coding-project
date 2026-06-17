"""上下文压缩器单元测试."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ai_coding.agent.context_compressor import ContextCompressor


def _make_read_file_observation(path: str, size: int = 500) -> ToolMessage:
    """构造一个模拟 read_file 返回的长 ToolMessage."""
    content = f"文件: {path}\n{'=' * 50}\n" + "\n".join(
        f"{i:4d} | line content for {path} index {i} " + "x" * 80
        for i in range(1, size + 1)
    ) + f"\n{'=' * 50}\n"
    return ToolMessage(content=content, tool_call_id=f"call_{path}")


class TestContextCompressor:
    def test_no_compress_when_under_budget(self):
        compressor = ContextCompressor(token_budget=10000)
        messages = [
            SystemMessage(content="system prompt"),
            HumanMessage(content="hi"),
            AIMessage(content="hello"),
        ]
        result = compressor.compress(messages)

        assert len(result) == len(messages)
        assert result[0].content == "system prompt"

    def test_compress_reduces_token_count(self):
        compressor = ContextCompressor(token_budget=2000)
        messages = [
            SystemMessage(content="system prompt"),
            HumanMessage(content="question 1"),
            AIMessage(content="answer 1"),
            _make_read_file_observation("a.py", size=200),
            HumanMessage(content="question 2"),
            AIMessage(content="answer 2"),
            _make_read_file_observation("b.py", size=200),
        ]

        original_tokens = compressor._estimate_tokens(messages)
        result = compressor.compress(messages)
        new_tokens = compressor._estimate_tokens(result)

        assert new_tokens < original_tokens
        # 最新一轮应被保留
        assert any("question 2" in str(m.content) for m in result)

    def test_deduplicate_read_file(self):
        compressor = ContextCompressor(token_budget=2000)
        messages = [
            SystemMessage(content="system"),
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            _make_read_file_observation("same.py", size=100),
            HumanMessage(content="q2"),
            AIMessage(content="a2"),
            _make_read_file_observation("same.py", size=100),
        ]

        result = compressor.compress(messages)
        # 即使被压缩，也不应报错；token 应下降
        assert compressor._estimate_tokens(result) < compressor._estimate_tokens(messages)

    def test_empty_messages(self):
        compressor = ContextCompressor(token_budget=1000)
        assert compressor.compress([]) == []
