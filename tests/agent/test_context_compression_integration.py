"""上下文压缩集成测试.

验证 ContextCompressor 的各项压缩策略以及 LangGraphAgent._maybe_compact()
的自动触发行为。
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ai_coding.agent.context_compressor import ContextCompressor
from ai_coding.agent.core import LangGraphAgent
from ai_coding.mock_llm import MockChatModel, mock_text


class TestContextCompressorStrategies:
    """直接测试 ContextCompressor 的策略."""

    def test_no_compress_when_under_budget(self) -> None:
        """预算内应原样返回."""
        compressor = ContextCompressor(token_budget=1000)
        messages = [
            SystemMessage(content="系统提示"),
            HumanMessage(content="你好"),
            AIMessage(content="你好！"),
        ]
        compressed, stats = compressor.compress_with_stats(messages)

        assert len(compressed) == 3
        assert stats["strategies_applied"] == []
        assert stats["original_tokens"] == stats["compressed_tokens"]

    def test_summarize_long_tool_message(self) -> None:
        """超长 ToolMessage 应被摘要."""
        compressor = ContextCompressor(token_budget=500, keep_recent_turns=1)
        long_content = "文件: src/main.py\n" + "\n".join(f"line {i}: code" for i in range(500))
        messages = [
            SystemMessage(content="系统提示"),
            HumanMessage(content="读取文件"),
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file", "args": {"path": "src/main.py"}, "id": "tc1"}],
            ),
            ToolMessage(content=long_content, tool_call_id="tc1"),
            HumanMessage(content="还有呢？"),
            AIMessage(content="读完了。"),
        ]
        compressed, stats = compressor.compress_with_stats(messages)

        assert "summarize" in stats["strategies_applied"]
        assert stats["compressed_tokens"] <= stats["original_tokens"]
        assert stats["compressed_tokens"] <= compressor.token_budget

    def test_deduplicate_file_reads(self) -> None:
        """同一文件多次读取应去重."""
        compressor = ContextCompressor(token_budget=200, keep_recent_turns=1)
        content = "文件: src/main.py\n" + "\n".join(f"line {i}" for i in range(100))
        messages = [
            SystemMessage(content="系统提示"),
            HumanMessage(content="读文件"),
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file", "args": {"path": "src/main.py"}, "id": "tc1"}],
            ),
            ToolMessage(content=content, tool_call_id="tc1"),
            HumanMessage(content="再读一次"),
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file", "args": {"path": "src/main.py"}, "id": "tc2"}],
            ),
            ToolMessage(content=content, tool_call_id="tc2"),
            HumanMessage(content="总结"),
            AIMessage(content="完成了。"),
        ]
        compressed, stats = compressor.compress_with_stats(messages)

        assert "dedup" in stats["strategies_applied"]
        assert stats["compressed_count"] < stats["original_count"]

    def test_keep_recent_turns(self) -> None:
        """最近 N 轮应完整保留，older turns 被摘要."""
        compressor = ContextCompressor(token_budget=40, keep_recent_turns=2)
        messages = [SystemMessage(content="系统提示")]
        for i in range(5):
            messages.append(HumanMessage(content=f"问题 {i}"))
            messages.append(AIMessage(content=f"回答 {i}"))

        compressed, stats = compressor.compress_with_stats(messages)

        assert "turns_summary" in stats["strategies_applied"]
        assert stats["compressed_count"] < stats["original_count"]
        # 最近 2 轮保留
        assert any(
            isinstance(m, HumanMessage) and m.content == "问题 3" for m in compressed
        )
        assert any(
            isinstance(m, HumanMessage) and m.content == "问题 4" for m in compressed
        )


class TestLangGraphAgentAutoCompact:
    """测试 LangGraphAgent 的自动压缩触发."""

    def test_auto_compact_triggered(self) -> None:
        """当上下文超过阈值时，_maybe_compact 应触发压缩."""
        llm = MockChatModel(responses=[mock_text("ok")])
        agent = LangGraphAgent(
            work_dir=".",
            llm=llm,
            tools=[],
            system_prompt="测试",
            short_term_memory_budget=500,
        )

        messages = [SystemMessage(content="系统提示")]
        for i in range(20):
            messages.append(HumanMessage(content=f"问题 {i}: " + "x" * 200))
            messages.append(AIMessage(content=f"回答 {i}: " + "y" * 200))

        agent.state = {
            "messages": messages,
            "file_snapshots": {},
            "todos": [],
            "globally_approved_tools": [],
            "background_tasks": [],
            "plan_mode": False,
            "plan_file_path": "",
            "sub_agents": [],
        }
        original_count = len(agent.state["messages"])

        agent._maybe_compact()

        compressed_count = len(agent.state["messages"])
        assert compressed_count < original_count
        # 压缩后应在预算内（考虑 AUTO_COMPACT_THRESHOLD=0.95）
        compressor = agent.context_compressor
        assert compressor._estimate_tokens(agent.state["messages"]) <= compressor.token_budget
