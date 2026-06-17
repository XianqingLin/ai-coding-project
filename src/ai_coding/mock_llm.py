"""Mock LLM 实现.

用于离线测试 Agent 逻辑，不消耗真实 API。
支持预设响应序列和交互式调试模式。
"""

from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage

from ai_coding.config import MOCK_INTERACTIVE
from ai_coding.logger import get_logger

logger = get_logger(__name__)


def mock_tool_call(name: str, args: Dict[str, Any], content: str = "", call_id: str = "") -> AIMessage:
    """快速构造一个带 tool_calls 的 AIMessage.

    Args:
        name: 工具名.
        args: 工具参数.
        content: AI 的思考过程文本.
        call_id: tool_call ID, 为空时自动生成.

    Returns:
        AIMessage 实例.

    """
    return AIMessage(
        content=content,
        tool_calls=[{
            "name": name,
            "args": args,
            "id": call_id or f"mock_{name}",
            "type": "tool_call",
        }],
        additional_kwargs={"reasoning_content": ""},
    )


def mock_text(text: str) -> AIMessage:
    """快速构造一个纯文本 AIMessage."""
    return AIMessage(content=text)


class MockChatModel:
    """模拟 LLM，兼容 LangChain ChatModel 接口.

    使用示例:
        >>> from ai_coding.mock_llm import MockChatModel, mock_tool_call, mock_text
        >>> from ai_coding.tools import create_default_tools
        >>> llm = MockChatModel(responses=[
        ...     mock_tool_call("list_dir", {"path": "."}, content="看看目录结构"),
        ...     mock_tool_call("read_file", {"path": "main.py"}, content="读一下主文件"),
        ...     mock_text("任务完成！"),
        ... ])
        >>> from ai_coding.agent.core import LangGraphAgent
        >>> agent = LangGraphAgent(llm=llm, tools=create_default_tools())
        >>> result = agent.run("帮我看看项目")

    """

    def __init__(
        self,
        model_name: str = "mock",
        responses: Optional[List[AIMessage]] = None,
        interactive: bool = MOCK_INTERACTIVE,
    ) -> None:
        """初始化 Mock LLM.

        Args:
            model_name: 模型名称标识.
            responses: 预设的 AIMessage 回复序列.
            interactive: 是否启用交互模式（每轮暂停等待人工输入）.

        """
        self.model_name = model_name
        self.responses = responses or []
        self.interactive = interactive
        self._index = 0
        self._tools: List[Any] = []

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "MockChatModel":
        """绑定工具（与真实 LLM 接口兼容）.

        返回 self，因为 Mock LLM 不需要真正的工具绑定.

        """
        self._tools = tools
        return self

    def invoke(self, messages: List[BaseMessage], **kwargs: Any) -> AIMessage:
        """模拟 LLM 调用，返回预设回复或交互输入.

        Args:
            messages: 当前消息列表（仅用于日志/显示）.

        Returns:
            AIMessage 实例.

        """
        last_msg = messages[-1] if messages else None
        last_content = last_msg.content[:150] if last_msg and hasattr(last_msg, "content") else ""
        logger.info(f"[MockLLM] invoke | index={self._index}/{len(self.responses)} | last_msg={last_content!r}")

        if self.interactive:
            return self._interactive_invoke(messages)

        if self._index < len(self.responses):
            resp = self.responses[self._index]
            self._index += 1
            if isinstance(resp, str):
                resp = AIMessage(content=resp)
            # 确保有 reasoning_content 字段（兼容 Kimi K2.6 要求）
            if resp.tool_calls and "reasoning_content" not in resp.additional_kwargs:
                resp.additional_kwargs["reasoning_content"] = ""
            logger.info(f"[MockLLM] -> preset response: content_len={len(resp.content)} tool_calls={[tc.get('name') for tc in resp.tool_calls]}")
            return resp

        # 预设回复用完后的默认行为
        logger.warning("[MockLLM] 预设回复已用完，返回默认消息")
        return AIMessage(content="[MockLLM] 预设回复序列已耗尽。请增加 responses 列表长度。")

    def stream(self, messages: List[BaseMessage], **kwargs: Any):
        """模拟流式输出.

        目前直接返回完整消息（非真正的逐字流式）.

        """
        msg = self.invoke(messages)
        if msg.content:
            yield msg.content

    def _interactive_invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """交互模式：暂停并等待人工输入模拟回复."""
        last_msg = messages[-1] if messages else None
        last_content = last_msg.content[:200] if last_msg and hasattr(last_msg, "content") else ""

        print("\n" + "=" * 50)
        print(f"[MockLLM 交互模式] 第 {self._index + 1} 轮")
        print(f"  最后消息: {last_content!r}")
        print(f"  可用工具: {[getattr(t, 'name', str(t)) for t in self._tools]}")
        print("-" * 50)
        print("输入格式:")
        print("  text:<内容>          -> 纯文本回复")
        print("  tool:<工具名> <JSON>  -> 调用工具（如 tool:read_file {'path': 'x.py'}）")
        print("  tool:<工具名>         -> 调用工具（无参数）")
        print("=" * 50)

        user_input = input("[MockLLM] 你的输入: ").strip()

        if user_input.startswith("text:"):
            text = user_input[5:].strip()
            return AIMessage(content=text)

        if user_input.startswith("tool:"):
            rest = user_input[5:].strip()
            parts = rest.split(" ", 1)
            tool_name = parts[0]
            args = {}
            if len(parts) > 1:
                import json
                try:
                    args = json.loads(parts[1])
                except json.JSONDecodeError:
                    print("[MockLLM] 警告: 无法解析 JSON 参数，使用空参数")
            return mock_tool_call(tool_name, args)

        # 默认当作纯文本
        return AIMessage(content=user_input)
