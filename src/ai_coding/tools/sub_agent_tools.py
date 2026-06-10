"""子 Agent 相关工具.

提供 dispatch_sub_agent、list_sub_agents、get_sub_agent_result 三个工具.
"""

import time
from typing import Any, Callable, List, Optional

from ai_coding.tools.base import Tool, ToolParameter


class DispatchSubAgentTool(Tool):
    """派发子 Agent 执行任务."""

    name = "dispatch_sub_agent"
    requires_approval = True
    description = (
        "向子 Agent 派发一个独立任务。子 Agent 拥有独立的上下文窗口和工具集，"
        "不会污染主 Agent 的对话历史。\n"
        "内置三种子 Agent：\n"
        "- coder（默认）：通用软件工程助手，可读写文件、执行命令\n"
        "- explore：只读探索，适合快速梳理代码库\n"
        "- plan：只读+规划，专注于架构设计，不执行命令\n\n"
        "同步模式直接返回结果；后台模式返回任务 ID，完成后结果自动回流到主上下文。\n"
        "提供 instance_id 可唤回已有实例继续推进。"
    )

    def __init__(self) -> None:
        super().__init__()
        self._llm: Any = None
        self._llm_factory: Optional[Callable[[], Any]] = None
        from ai_coding.sub_agent_manager import SubAgentManager
        self._manager = SubAgentManager()

    def set_llm(self, llm: Any, llm_factory: Optional[Callable[[], Any]] = None) -> None:
        """由 LangGraphAgent 注入 LLM 依赖."""
        self._llm = llm
        self._llm_factory = llm_factory

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "agent_type",
                "string",
                "子 Agent 类型：coder（默认）、explore（只读）、plan（只规划）",
                required=False,
                default="coder",
                enum=["coder", "explore", "plan"],
            ),
            ToolParameter("task", "string", "要派发给子 Agent 的任务描述"),
            ToolParameter(
                "run_in_background",
                "boolean",
                "是否后台运行，默认 false（同步模式直接返回结果）",
                required=False,
                default=False,
            ),
            ToolParameter(
                "instance_id",
                "string",
                "可选。提供已有实例 ID 以唤回该实例继续任务",
                required=False,
            ),
        ]

    def execute(
        self,
        agent_type: str = "coder",
        task: str = "",
        run_in_background: bool = False,
        instance_id: str = "",
    ) -> str:
        if not task:
            return "[错误] task 参数不能为空"

        if agent_type not in ("coder", "explore", "plan"):
            return f"[错误] 不支持的 agent_type: {agent_type!r}"

        if self._llm is None and self._llm_factory is None:
            return "[错误] 子 Agent 工具未初始化 LLM，无法派发"

        return self._manager.dispatch(
            agent_type=agent_type,
            task=task,
            llm=self._llm,
            llm_factory=self._llm_factory,
            run_in_background=run_in_background,
            instance_id=instance_id or None,
        )


class ListSubAgentsTool(Tool):
    """列出所有子 Agent 实例."""

    name = "list_sub_agents"
    requires_approval = False
    description = "列出所有子 Agent 实例及其状态，可按状态过滤。"

    def __init__(self) -> None:
        super().__init__()
        from ai_coding.sub_agent_manager import SubAgentManager
        self._manager = SubAgentManager()

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "status",
                "string",
                "按状态过滤：running / completed / failed",
                required=False,
                enum=["running", "completed", "failed"],
            ),
        ]

    def execute(self, status: str = "") -> str:
        instances = self._manager.list_instances(status=status or None)
        if not instances:
            scope = f" ({status})" if status else ""
            return f"当前没有{scope}子 Agent 实例。"

        lines = [f"子 Agent 实例 ({len(instances)} 个):"]
        for inst in instances:
            duration = ""
            if inst.completed_at and inst.created_at:
                duration = f" | 耗时 {inst.completed_at - inst.created_at:.1f}s"
            lines.append(
                f"  [{inst.status:10s}] {inst.instance_id} ({inst.agent_type})"
                f"{duration}\n    任务: {inst.task[:80]}"
            )
        return "\n".join(lines)


class GetSubAgentResultTool(Tool):
    """获取子 Agent 结果."""

    name = "get_sub_agent_result"
    requires_approval = False
    description = (
        "获取指定子 Agent 实例的结果。\n"
        "block=true 时会阻塞等待任务完成（最多等待 timeout 秒）。"
    )

    def __init__(self) -> None:
        super().__init__()
        from ai_coding.sub_agent_manager import SubAgentManager
        self._manager = SubAgentManager()

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("instance_id", "string", "子 Agent 实例 ID"),
            ToolParameter(
                "block",
                "boolean",
                "是否阻塞等待完成，默认 false",
                required=False,
                default=False,
            ),
            ToolParameter(
                "timeout",
                "integer",
                "等待秒数，默认 30，范围 1-300",
                required=False,
                default=30,
            ),
        ]

    def execute(self, instance_id: str, block: bool = False, timeout: int = 30) -> str:
        timeout = max(1, min(300, int(timeout)))

        if block:
            start = time.time()
            while time.time() - start < timeout:
                inst = self._manager.get_instance(instance_id)
                if not inst:
                    return f"[错误] 实例不存在: {instance_id}"
                if inst.status != "running":
                    break
                time.sleep(0.5)

        inst = self._manager.get_instance(instance_id)
        if not inst:
            return f"[错误] 实例不存在: {instance_id}"

        status = inst.status
        header = (
            f"[SubAgent] {instance_id}\n"
            f"[Status] {status}\n"
            f"[Type] {inst.agent_type}\n"
            f"[Task] {inst.task}\n"
        )
        if status == "running":
            header += "[Result] 任务仍在运行中，可稍后再次查询或设置 block=true 等待。"
            return header

        header += f"\n[Result]\n{'='*50}\n{inst.result}\n{'='*50}"
        return header
