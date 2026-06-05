"""Agent 节点工厂.

负责调用 LLM 前的上下文准备、LLM 调用、日志记录.
"""

import time
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState

from ai_coding.logger import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

    from ai_coding.agent.context import ContextManager
    from ai_coding.agent.phase import PhaseController

logger = get_logger(__name__)


def create_agent_node(
    llm: "BaseChatModel",
    context_mgr: "ContextManager",
    phase_ctrl: "PhaseController",
):
    """创建 Agent 节点函数.

    Args:
        llm: 已绑定工具的 LangChain ChatModel.
        context_mgr: 上下文管理器，负责滑动窗口、压缩、系统提示注入.
        phase_ctrl: 阶段控制器，用于日志中的阶段标签.

    Returns:
        符合 LangGraph 节点签名的 callable.
    """

    def agent_node(state: MessagesState):
        """Agent 节点：准备上下文，调用 LLM，返回 AI 消息."""
        messages = list(state["messages"])

        # 计算当前步数（已完成的 AI 决策轮数）
        current_step = len([m for m in messages if isinstance(m, AIMessage)])

        # 上下文预处理（系统提示 + 滑动窗口 + 压缩）
        messages = context_mgr.prepare(messages)

        # 阶段标签（仅用于日志）
        explore_steps = phase_ctrl.count_explore_steps(messages)
        phase_tag = (
            f"EDIT(plan) explore={explore_steps}"
            if phase_ctrl.has_plan
            else f"EXPLORE({explore_steps})"
        )

        # 记录 LLM 输入摘要
        last_msg = messages[-1] if messages else None
        last_content = (
            last_msg.content[:200]
            if last_msg and hasattr(last_msg, "content")
            else ""
        )
        logger.info(
            f"[LLM IN]  messages={len(messages)} step={current_step + 1} "
            f"phase={phase_tag} last_content={last_content[:100]!r}"
        )
        t0 = time.time()

        # 调用 LLM
        response = llm.invoke(messages)

        elapsed = time.time() - t0
        if isinstance(response, AIMessage):
            tool_names = [tc.get("name", "") for tc in response.tool_calls]
            reasoning = response.additional_kwargs.get("reasoning_content", "")
            logger.info(
                f"[LLM OUT] elapsed={elapsed:.2f}s "
                f"content_len={len(response.content)} "
                f"tool_calls={tool_names} "
                f"reasoning_len={len(reasoning) if reasoning else 0}"
            )
            logger.debug(f"[LLM OUT] content={response.content[:500]!r}")

        return {"messages": [response]}

    return agent_node
