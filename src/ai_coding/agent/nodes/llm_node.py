"""LLM 节点工厂.

负责注入最新的文件快照上下文后调用 LLM.
历史压缩已由 AgentState 自我管理，此节点不再处理压缩逻辑.
"""

import time
from typing import TYPE_CHECKING, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from ai_coding.agent.state import AgentState
from ai_coding.sub_agent_manager import SubAgentManager
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


def _build_sub_agent_human_message(manager: SubAgentManager) -> HumanMessage:
    """将已完成的子 Agent 结果格式化为合成 HumanMessage."""
    pending = manager.get_pending_notifications()
    if not pending:
        return HumanMessage(content="")

    lines = []
    for inst in pending:
        status_label = "完成" if inst.status == "completed" else "失败"
        lines.append(
            f"[子 Agent {status_label}] {inst.instance_id} ({inst.agent_type})"
        )
        lines.append(f"任务: {inst.task}")
        result_preview = inst.result[:2000]
        if len(inst.result) > 2000:
            result_preview += "\n... (结果已截断)"
        lines.append(f"结果:\n{result_preview}")

    return HumanMessage(content="\n".join(lines))


def _build_todo_context_message(todos: List[dict]) -> SystemMessage:
    """将任务列表格式化为 LLM 上下文消息."""
    if not todos:
        return SystemMessage(content="")

    lines = ["## 当前任务列表"]
    done_count = sum(1 for t in todos if t.get("done"))
    for i, t in enumerate(todos, 1):
        mark = "[x]" if t.get("done") else "[ ]"
        lines.append(f"  {mark} {i}. {t['task']}")
    lines.append(f"\n进度: {done_count}/{len(todos)} 已完成")
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
        todo_ctx_msg = _build_todo_context_message(state.get("todos", []))

        insert_idx = 0
        for i, m in enumerate(messages):
            if isinstance(m, SystemMessage):
                insert_idx = i + 1
                break

        if file_ctx_msg.content:
            messages.insert(insert_idx, file_ctx_msg)
            insert_idx += 1
            logger.debug(f"[LLM] 注入文件快照 | 文件数={len(state.get('file_snapshots', {}))}")

        if todo_ctx_msg.content:
            messages.insert(insert_idx, todo_ctx_msg)
            insert_idx += 1
            logger.debug(f"[LLM] 注入任务列表 | 任务数={len(state.get('todos', []))}")

        # 注入已完成的子 Agent 结果（作为合成 HumanMessage 追加到末尾）
        manager = SubAgentManager()
        sub_agent_human_msg = _build_sub_agent_human_message(manager)
        if sub_agent_human_msg.content:
            messages.append(sub_agent_human_msg)
            logger.info(f"[LLM] 注入子 Agent 结果 HumanMessage | 数量={len(manager.get_pending_notifications())}")

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

        # 标记已通知的子 Agent
        updated_sub_agents = list(state.get("sub_agents", []))
        if sub_agent_human_msg.content:
            pending = manager.get_pending_notifications()
            for inst in pending:
                manager.mark_notified(inst.instance_id)
            # 同步 manager 状态到 state 列表
            updated_sub_agents = [manager.to_dict(i) for i in manager.list_instances()]

        return {
            "messages": [response],
            "sub_agents": updated_sub_agents,
        }

    return llm_node
