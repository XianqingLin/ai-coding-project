"""Todo 工具.

Agent 用于自我管理子任务进度，提升复杂任务的可观测性.
"""

from typing import List, Optional

from ai_coding.tools.base import Tool, ToolParameter


class TodoTool(Tool):
    """任务列表管理工具."""

    name = "set_todo"
    description = (
        "创建、更新、删除或查看任务列表，帮助跟踪复杂任务的子任务进度。\n"
        "你可以在探索阶段分解任务，在修改阶段标记完成进度。\n"
        "示例：\n"
        "  - action=add, task='理解 require() 的实现'\n"
        "  - action=complete, index=1\n"
        "  - action=remove, index=2\n"
        "  - action=update, index=1, task='修改后的描述'\n"
        "  - action=list"
    )

    def __init__(self) -> None:
        super().__init__()
        self._todos: List[dict] = []

    def sync(self, todos: List[dict]) -> None:
        """从外部状态同步 todo 列表（如 AgentState 中的 todos）."""
        self._todos = [dict(t) for t in todos]

    @property
    def todos(self) -> List[dict]:
        """返回当前任务列表副本."""
        return [dict(t) for t in self._todos]

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "action",
                "string",
                "操作类型",
                enum=["add", "complete", "remove", "update", "list"],
            ),
            ToolParameter(
                "task",
                "string",
                "任务描述（action=add 时必填，action=complete 时可作为替代索引的匹配文本）",
                required=False,
                default="",
            ),
            ToolParameter(
                "index",
                "integer",
                "任务序号，从 1 开始（action=complete / remove / update 时使用）",
                required=False,
                default=None,
            ),
        ]

    def execute(  # type: ignore[override]
        self, action: str, task: str = "", index: Optional[int] = None
    ) -> str:
        action = action.lower().strip()

        if action == "add":
            if not task:
                return "[错误] action=add 时必须提供 task 参数"
            self._todos.append({"task": task, "done": False})
            return f"[待办] 已添加: {task} (当前共 {len(self._todos)} 项)"

        if action == "complete":
            return self._mark_done(task=task, index=index)

        if action == "remove":
            return self._remove(task=task, index=index)

        if action == "update":
            return self._update(new_task=task, index=index)

        if action == "list":
            if not self._todos:
                return "当前无待办任务。"
            lines = ["任务列表:"]
            done_count = sum(1 for t in self._todos if t["done"])
            for i, t in enumerate(self._todos, 1):
                mark = "[x]" if t["done"] else "[ ]"
                lines.append(f"  {mark} {i}. {t['task']}")
            lines.append(f"\n进度: {done_count}/{len(self._todos)} 已完成")
            return "\n".join(lines)

        return f"[错误] 未知的 action: {action}"

    def _mark_done(self, task: str = "", index: Optional[int] = None) -> str:
        """标记任务为已完成."""
        if index is not None:
            idx = index - 1
            if 0 <= idx < len(self._todos):
                self._todos[idx]["done"] = True
                return f"[完成] {self._todos[idx]['task']}"
            return f"[错误] 序号 {index} 超出范围 (1-{len(self._todos)})"

        if task:
            for i, t in enumerate(self._todos):
                if not t["done"] and task.lower() in t["task"].lower():
                    self._todos[i]["done"] = True
                    return f"[完成] {t['task']}"
            return f"[错误] 未找到匹配的任务: {task}"

        return "[错误] action=complete 时需要提供 index 或 task 参数"

    def _remove(self, task: str = "", index: Optional[int] = None) -> str:
        """删除任务."""
        if index is not None:
            idx = index - 1
            if 0 <= idx < len(self._todos):
                removed = self._todos.pop(idx)
                return f"[删除] {removed['task']}"
            return f"[错误] 序号 {index} 超出范围 (1-{len(self._todos)})"

        if task:
            for i, t in enumerate(self._todos):
                if task.lower() in t["task"].lower():
                    removed = self._todos.pop(i)
                    return f"[删除] {removed['task']}"
            return f"[错误] 未找到匹配的任务: {task}"

        return "[错误] action=remove 时需要提供 index 或 task 参数"

    def _update(self, new_task: str = "", index: Optional[int] = None) -> str:
        """修改任务描述."""
        if not new_task:
            return "[错误] action=update 时必须提供 task 参数作为新描述"

        if index is not None:
            idx = index - 1
            if 0 <= idx < len(self._todos):
                old = self._todos[idx]["task"]
                self._todos[idx]["task"] = new_task
                return f"[更新] {old} → {new_task}"
            return f"[错误] 序号 {index} 超出范围 (1-{len(self._todos)})"

        if new_task:
            # update 不支持纯文本匹配（容易误操作），必须提供 index
            return "[错误] action=update 时必须提供 index 参数指定要修改的任务"

        return "[错误] action=update 时需要提供 index 和 task 参数"
