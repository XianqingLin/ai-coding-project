"""LLM 节点工厂.

负责注入最新的文件快照上下文后调用 LLM.
历史压缩已由 AgentState 自我管理，此节点不再处理压缩逻辑.
"""

import time
from typing import TYPE_CHECKING, Dict

from langchain_core.messages import SystemMessage

from ai_coding.agent.state import AgentState
from ai_coding.logger import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = get_logger(__name__)

# 单个文件快照在上下文中最大保留行数
FILE_SNAPSHOT_MAX_LINES = 50


def _build_file_context_message(snapshots: Dict[str, str]) -> SystemMessage:
    """将文件快照格式化为 LLM 上下文消息.

    对超大文件做截断，只保留首尾各一部分.
    """
    if not snapshots:
        return SystemMessage(content="")

    lines = ["## 当前文件快照（最新状态）"]
    for path, content in snapshots.items():
        lines.append(f"\n--- {path} ---")
        content_lines = content.split("\n")
        total = len(content_lines)
        if total > FILE_SNAPSHOT_MAX_LINES:
            head = "\n".join(content_lines[: FILE_SNAPSHOT_MAX_LINES // 2])
            tail = "\n".join(content_lines[-FILE_SNAPSHOT_MAX_LINES // 2 :])
            lines.append(f"{head}\n...（省略 {total - FILE_SNAPSHOT_MAX_LINES} 行）...\n{tail}")
        else:
            lines.append(content)

    return SystemMessage(content="\n".join(lines))


def create_llm_node(llm: "BaseChatModel"):
    """创建 LLM 节点函数.

    Args:
        llm: 已绑定工具的 LangChain ChatModel.

    Returns:
        符合 LangGraph 节点签名的 callable.
    """

    def llm_node(state: AgentState):
        """LLM 节点：注入文件快照后调用 LLM."""
        messages = list(state["messages"])

        # 注入文件快照上下文（插入到 SystemMessage 之后）
        file_ctx_msg = _build_file_context_message(state.get("file_snapshots", {}))
        if file_ctx_msg.content:
            insert_idx = 0
            for i, m in enumerate(messages):
                if isinstance(m, SystemMessage):
                    insert_idx = i + 1
                    break
            messages.insert(insert_idx, file_ctx_msg)
            logger.debug(f"[LLM] 注入文件快照 | 文件数={len(state.get('file_snapshots', {}))}")

        # 记录 LLM 输入摘要
        last_msg = messages[-1] if messages else None
        last_content = (
            last_msg.content[:200]
            if last_msg and hasattr(last_msg, "content")
            else ""
        )
        logger.info(
            f"[LLM IN]  messages={len(messages)} "
            f"last_content={last_content[:100]!r}"
        )
        t0 = time.time()

        # 调用 LLM
        response = llm.invoke(messages)

        elapsed = time.time() - t0
        if hasattr(response, "tool_calls"):
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

    return llm_node
