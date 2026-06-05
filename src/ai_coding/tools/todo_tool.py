"""Todo 工具.

Agent 用于自我管理子任务进度，提升复杂任务的可观测性.
"""

from typing import List, Optional

from ai_coding.tools.base import Tool, ToolParameter


class TodoTool(Tool):
    """任务列表管理工具."""

    name = "set_todo"
    description = (
        "创建、更新或查看任务列表，帮助跟踪复杂任务的子任务进度。\n"
        "你可以在探索阶段分解任务，在修改阶段标记完成进度。\n"
        "示例：\n"
        "  - action=add, task='理解 require() 的实现'\n"
        "  - action=complete, index=1\n"
        "  - action=list"
    )

    def __init__(self) -> None:
        super().__init__()
        self._todos: List[dict] = []

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "action",
                "string",
                "操作类型",
                enum=["add", "complete", "list"],
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
                "任务序号，从 1 开始（action=complete 时使用）",
                required=False,
                default=None,
            ),
        ]

    def execute(self, action: str, task: str = "", index: Optional[int] = None) -> str:
        action = action.lower().strip()

        if action == "add":
            if not task:
                return "[错误] action=add 时必须提供 task 参数"
            self._todos.append({"task": task, "done": False})
            return f"[待办] 已添加: {task} (当前共 {len(self._todos)} 项)"

        if action == "complete":
            if index is not None:
                idx = index - 1  # 用户传入 1-based
                if 0 <= idx < len(self._todos):
                    self._todos[idx]["done"] = True
                    return f"[完成] {self._todos[idx]['task']}"
                return f"[错误] 序号 {index} 超出范围 (1-{len(self._todos)})"

            if task:
                # 通过文本匹配
                for i, t in enumerate(self._todos):
                    if not t["done"] and task.lower() in t["task"].lower():
                        self._todos[i]["done"] = True
                        return f"[完成] {t['task']}"
                return f"[错误] 未找到匹配的任务: {task}"

            return "[错误] action=complete 时需要提供 index 或 task 参数"

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

        return f"[错误] 未知 action: {action}. 可选: add, complete, list"
