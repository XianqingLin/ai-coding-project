"""上下文管理模块.

负责对话历史的滑动窗口截断、消息压缩、系统提示注入.
与具体 Agent 业务无关，可复用于任何长对话场景.
"""

from typing import List

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ai_coding.logger import get_logger

logger = get_logger(__name__)


class ContextManager:
    """对话上下文管理器.

    在调用 LLM 前对消息列表进行预处理：
    1. 注入系统提示（首轮时）
    2. 滑动窗口截断（保留最近 N 轮完整对话）
    3. 消息压缩（截断超长 ToolMessage）

    Attributes:
        system_prompt: 系统提示文本. 为 None 时不注入.
        max_turns: 滑动窗口保留的最近对话轮数.
        compress_threshold: 触发压缩的字符数阈值.
    """

    def __init__(
        self,
        system_prompt: str = "",
        max_turns: int = 8,
        compress_threshold: int = 40000,
    ) -> None:
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.compress_threshold = compress_threshold

    def prepare(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """预处理消息列表，返回适合传给 LLM 的消息.

        处理顺序：
        1. 注入系统提示（只在首轮对话时）
        2. 滑动窗口截断
        3. 消息压缩（如果截断后仍然过长）

        Args:
            messages: 原始消息列表（来自 MessagesState）.

        Returns:
            预处理后的消息列表.
        """
        msgs = list(messages)

        # 1. 系统提示注入
        msgs = self._inject_system_prompt(msgs)

        # 2. 滑动窗口截断
        msgs = self._apply_sliding_window(msgs)

        # 3. 消息压缩
        total_chars = sum(len(m.content) for m in msgs if hasattr(m, "content"))
        if total_chars > self.compress_threshold:
            msgs = self._compress_messages(msgs)

        return msgs

    def _inject_system_prompt(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """在首轮对话时注入系统提示.

        只在没有任何 AI 决策或工具执行记录时注入，避免重复.
        """
        if not self.system_prompt:
            return messages
        if any(isinstance(m, (AIMessage, ToolMessage)) for m in messages):
            return messages
        return [SystemMessage(content=self.system_prompt)] + messages

    def _apply_sliding_window(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """滑动窗口截断：保留最近 N 轮完整 agent-tool 对话，更早的折叠为摘要.

        关键约束：一个 AIMessage 可能触发多个 ToolMessage（如同时调用
        多个 read_file），截断时必须保留完整的 "AIMessage + 对应的所有
        ToolMessage"，否则会出现 tool_call_id 不匹配导致 API 400 错误.

        Checkpoint 中的完整历史不受影响，仅影响传给 LLM 的上下文窗口.
        """
        preserved: List[BaseMessage] = []
        conversation: List[BaseMessage] = []

        for m in messages:
            if isinstance(m, (SystemMessage, HumanMessage)):
                preserved.append(m)
            else:
                conversation.append(m)

        # 从后往前数完整的轮次
        # 一轮 = 1 个 AIMessage + 它对应的所有 ToolMessage
        kept: List[BaseMessage] = []
        turns = 0
        i = len(conversation) - 1

        while i >= 0 and turns < self.max_turns:
            # 1. 从后往前收集所有连续的 ToolMessage（属于当前轮）
            while i >= 0 and isinstance(conversation[i], ToolMessage):
                kept.insert(0, conversation[i])
                i -= 1

            # 2. 收集对应的 AIMessage
            if i >= 0 and isinstance(conversation[i], AIMessage):
                kept.insert(0, conversation[i])
                i -= 1
                turns += 1
            else:
                # 结构异常（可能是最终答案的 AIMessage，后面没有 ToolMessage）
                if i >= 0:
                    kept.insert(0, conversation[i])
                    i -= 1
                break

        if i >= 0:
            dropped = conversation[: i + 1]
            dropped_tool_calls = sum(
                1 for m in dropped if isinstance(m, AIMessage) and m.tool_calls
            )
            dropped_obs = sum(1 for m in dropped if isinstance(m, ToolMessage))

            summary = SystemMessage(
                content=(
                    f"[历史摘要] 前 {len(dropped)} 条消息已折叠 "
                    f"({dropped_tool_calls} 次工具调用, {dropped_obs} 次观察结果). "
                    f"保留最近 {turns} 轮完整上下文."
                )
            )
            result = preserved + [summary] + kept
            logger.info(
                f"[SLIDE] messages={len(messages)} -> {len(result)} "
                f"(保留最近 {turns} 轮, 折叠 {len(dropped)} 条)"
            )
            return result

        return messages

    def _compress_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """压缩历史消息中的大观察结果，防止上下文膨胀.

        将长度超过 3000 字符的 ToolMessage 截断为"前 600 + 后 400"的摘要形式，
        保留文件名和关键信息的同时大幅降低上下文大小.
        """
        compressed = []
        total_before = 0
        total_after = 0
        for m in messages:
            content = m.content if hasattr(m, "content") else ""
            total_before += len(content)
            if isinstance(m, ToolMessage) and len(content) > 3000:
                prefix = content[:600]
                suffix = content[-400:]
                summary = (
                    f"{prefix}\n"
                    f"\n... [内容已压缩，原长度 {len(content)} 字符] ...\n"
                    f"{suffix}"
                )
                compressed.append(ToolMessage(content=summary, tool_call_id=m.tool_call_id))
                total_after += len(summary)
            else:
                compressed.append(m)
                total_after += len(content)
        logger.info(
            f"[COMPRESS] before={total_before} after={total_after} "
            f"saved={total_before - total_after} "
            f"({(total_before - total_after) / max(total_before, 1) * 100:.1f}%)"
        )
        return compressed
