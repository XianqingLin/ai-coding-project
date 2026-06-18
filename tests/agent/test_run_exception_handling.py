"""Agent 运行异常分类处理测试.

验证 LangGraphAgent 对 RecursionError、InterruptedError、API/网络异常、
其他异常的分类提示行为。
"""

from typing import Any, Iterator, List

from langchain_core.messages import AIMessage, BaseMessage

from ai_coding.agent.core import LangGraphAgent
from ai_coding.mock_llm import MockChatModel, mock_text


class _FailingMockLLM(MockChatModel):
    """可控制抛出异常的 Mock LLM."""

    def __init__(self, exc: Exception) -> None:
        super().__init__(responses=[mock_text("should-not-reach")])
        self._exc = exc

    def invoke(self, messages: List[BaseMessage], **kwargs: Any) -> AIMessage:
        raise self._exc

    def stream(self, messages: List[BaseMessage], **kwargs: Any) -> Iterator[str]:
        raise self._exc


class _MockAPIError(Exception):
    """模拟 API 异常，类名包含 APIError 以便被启发式识别."""

    pass


class _MockGenericError(Exception):
    """模拟普通未知异常."""

    pass


class _MockConnectionError(Exception):
    """模拟网络连接异常."""

    pass


class TestRunExceptionClassification:
    """非流式运行异常分类测试."""

    def test_recursion_error_message(self):
        """RecursionError 应提示递归上限."""
        llm = _FailingMockLLM(RecursionError("maximum recursion depth exceeded"))
        agent = LangGraphAgent(work_dir=".", llm=llm, tools=[], system_prompt="test")

        result = agent.run("hello")

        assert "递归" in result or "迭代次数" in result
        assert "执行失败" not in result

    def test_api_error_message(self):
        """API 异常应提示网络/配置问题."""
        llm = _FailingMockLLM(_MockAPIError("rate limit exceeded"))
        agent = LangGraphAgent(work_dir=".", llm=llm, tools=[], system_prompt="test")

        result = agent.run("hello")

        assert "API" in result or "网络" in result or "配置" in result
        assert "执行失败" not in result

    def test_connection_error_message(self):
        """网络连接异常应提示网络/配置问题."""
        llm = _FailingMockLLM(_MockConnectionError("connection refused"))
        agent = LangGraphAgent(work_dir=".", llm=llm, tools=[], system_prompt="test")

        result = agent.run("hello")

        assert "API" in result or "网络" in result or "配置" in result
        assert "执行失败" not in result

    def test_generic_error_message(self):
        """普通异常保留默认提示."""
        llm = _FailingMockLLM(_MockGenericError("something went wrong"))
        agent = LangGraphAgent(work_dir=".", llm=llm, tools=[], system_prompt="test")

        result = agent.run("hello")

        assert "执行失败" in result


class TestHandleRunException:
    """直接测试 _handle_run_exception 分类逻辑."""

    def test_interrupted_error_message(self):
        """InterruptedError 应提示已取消."""
        agent = LangGraphAgent(
            work_dir=".", llm=MockChatModel(), tools=[], system_prompt="test"
        )

        msg = agent._handle_run_exception(InterruptedError("cancelled"), "run")

        assert "取消" in msg
        assert "执行失败" not in msg

    def test_recursion_error_direct(self):
        """直接调用时 RecursionError 分类正确."""
        agent = LangGraphAgent(
            work_dir=".", llm=MockChatModel(), tools=[], system_prompt="test"
        )

        msg = agent._handle_run_exception(RecursionError("depth"), "run_stream")

        assert "递归" in msg or "迭代次数" in msg

    def test_api_error_direct(self):
        """直接调用时 API 异常分类正确."""
        agent = LangGraphAgent(
            work_dir=".", llm=MockChatModel(), tools=[], system_prompt="test"
        )

        msg = agent._handle_run_exception(_MockAPIError("boom"), "run_stream_verbose")

        assert "API" in msg or "网络" in msg or "配置" in msg


class TestStreamExceptionClassification:
    """流式运行异常分类测试."""

    def test_stream_recursion_error_message(self):
        """流式 RecursionError 应提示递归上限."""
        llm = _FailingMockLLM(RecursionError("maximum recursion depth exceeded"))
        agent = LangGraphAgent(work_dir=".", llm=llm, tools=[], system_prompt="test")

        chunks = list(agent.run_stream("hello"))
        result = "".join(chunks)

        assert "递归" in result or "迭代次数" in result

    def test_stream_api_error_message(self):
        """流式 API 异常应提示网络/配置问题."""
        llm = _FailingMockLLM(_MockAPIError("connection timeout"))
        agent = LangGraphAgent(work_dir=".", llm=llm, tools=[], system_prompt="test")

        chunks = list(agent.run_stream("hello"))
        result = "".join(chunks)

        assert "API" in result or "网络" in result or "配置" in result

    def test_stream_generic_error_message(self):
        """流式普通异常保留默认提示."""
        llm = _FailingMockLLM(_MockGenericError("stream failed"))
        agent = LangGraphAgent(work_dir=".", llm=llm, tools=[], system_prompt="test")

        chunks = list(agent.run_stream("hello"))
        result = "".join(chunks)

        assert "执行失败" in result
