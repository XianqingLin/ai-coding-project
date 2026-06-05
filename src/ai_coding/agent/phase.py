"""阶段控制模块.

管理 Agent 的探索/修改两阶段流转，以及探索步数的软约束提示.
"""

from typing import List

from langchain_core.messages import AIMessage, BaseMessage

from ai_coding.logger import get_logger

logger = get_logger(__name__)


class PhaseController:
    """Agent 阶段控制器.

    跟踪 Agent 当前处于"探索阶段"还是"修改阶段"，
    并根据探索步数生成软约束提示，引导 Agent 尽快进入修改.

    Attributes:
        explore_tool_names: 被认定为"探索类"的工具名称集合.
    """

    EXPLORE_TOOL_NAMES = {"read_file", "list_dir", "grep"}

    def __init__(self) -> None:
        self.has_plan: bool = False
        self.plan_content: str = ""

    def on_plan_submitted(self, plan: str) -> None:
        """标记计划已提交，进入修改阶段."""
        self.has_plan = True
        self.plan_content = plan
        logger.info(f"[Plan] Agent 提交修改计划: {plan[:200]}")

    def reset(self) -> None:
        """重置阶段状态（开启新对话轮次时）."""
        self.has_plan = False
        self.plan_content = ""

    def count_explore_steps(self, messages: List[BaseMessage]) -> int:
        """统计历史消息中探索类工具的调用次数."""
        count = 0
        for m in messages:
            if not isinstance(m, AIMessage) or not m.tool_calls:
                continue
            names = {tc.get("name", "") for tc in m.tool_calls}
            if names & self.EXPLORE_TOOL_NAMES:
                count += 1
        return count

    def build_hint(self, explore_steps: int) -> str:
        """根据探索步数生成软约束提醒文本.

        措辞从建议逐步升级为要求，降低 LLM 的"自主选择"偏差.
        """
        if self.has_plan:
            return ""
        if explore_steps <= 2:
            return ""
        if explore_steps <= 5:
            return (
                f"[提示] 这是你第 {explore_steps} 次探索。"
                f"在读了 3-5 个关键文件后，你就应该已经足够了解代码。"
                f"现在必须调用 plan 工具提交修改计划，然后开始修改。"
            )
        if explore_steps <= 10:
            return (
                f"[要求] 你已探索 {explore_steps} 步，步数已过多。"
                f"无论你认为自己是否已完全理解代码，现在都必须调用 plan 工具。"
                f"plan 不需要完美，你可以在 str_replace_file 执行过程中调整。"
            )
        return (
            f"[强制] 你已探索 {explore_steps} 步，严重超时。"
            f"这是系统要求：你的下一步必须是调用 plan 工具提交修改计划。"
            f"不调用 plan 就无法开始修改，而你没有更多步数可以浪费在探索上。"
        )

    def build_post_explore_hint(self, tool_name: str, explore_steps: int) -> str:
        """为探索工具的执行结果追加提示.

        返回需要追加到工具结果后面的文本，若无需提示则返回空字符串.
        """
        if tool_name not in self.EXPLORE_TOOL_NAMES:
            return ""

        if self.has_plan:
            return (
                "\n\n[提醒] 你已提交修改计划，当前处于修改阶段。"
                "你的下一步必须是使用 str_replace_file 或 write_file 执行修改，"
                "而不是继续探索。请立即开始修改代码。"
            )

        hint = self.build_hint(explore_steps)
        return f"\n\n{hint}" if hint else ""
