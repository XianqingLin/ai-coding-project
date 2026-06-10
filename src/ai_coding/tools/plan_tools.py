"""Plan 模式工具.

提供 enter_plan_mode 和 exit_plan_mode 两个工具，用于进入和退出 Plan 模式。
Plan 模式下 Write/Edit 只允许操作计划文件，TaskStop 被拦截。
"""

import os
import time
from typing import Any, List

from ai_coding.tools.base import Tool, ToolParameter


# 保留词（case-insensitive）
_RESERVED_LABELS = {"approve", "reject", "reject and exit", "revise"}


class EnterPlanModeTool(Tool):
    """进入 Plan 模式."""

    name = "enter_plan_mode"
    requires_approval = False
    description = (
        "进入 Plan 模式。进入后只能使用 write_file 或 edit_file 修改计划文件，"
        "task_stop 工具将被拦截，其他工具仍按正常权限规则处理。"
        "完成计划后调用 exit_plan_mode 提交计划。"
    )

    @property
    def parameters(self) -> List[ToolParameter]:
        return []

    def execute(self) -> str:
        plan_dir = ".kimi/plans"
        os.makedirs(plan_dir, exist_ok=True)
        ts = int(time.time())
        plan_path = os.path.join(plan_dir, f"plan_{ts}.md")

        with open(plan_path, "w", encoding="utf-8") as f:
            f.write("# Plan\n\n")

        return (
            f"[成功] 已进入 Plan 模式。\n"
            f"计划文件路径: {plan_path}\n"
            f"工作流指引:\n"
            f"1. 只能使用 write_file 或 edit_file 修改计划文件 '{plan_path}'\n"
            f"2. task_stop 工具在 Plan 模式下不可用\n"
            f"3. 完成计划后调用 exit_plan_mode 提交计划\n"
            f"4. 其他工具（如 read_file, execute_command 等）仍可正常使用"
        )


class ExitPlanModeTool(Tool):
    """退出 Plan 模式并提交计划."""

    name = "exit_plan_mode"
    requires_approval = False
    description = (
        "退出 Plan 模式并提交计划。系统会读取当前计划文件内容并呈现给用户审批。\n"
        "可选参数 options 允许提供 1–3 个备选方案（每项为 dict，包含 label 和 description），"
        "供用户在审批时选择。label 最长 80 字符，不能重复，也不能使用保留词。"
    )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "options",
                "array",
                "备选方案列表，每项为包含 'label'(str) 和 'description'(str) 的字典，最多 3 项",
                required=False,
                default=[],
            ),
        ]

    def execute(self, options: List[Any] = None) -> str:
        validated = self._validate_options(options or [])
        if isinstance(validated, str) and validated.startswith("[错误]"):
            return validated
        return ""

    def _validate_options(self, options: List[Any]) -> str:
        """验证 options 格式，返回错误消息或空字符串表示通过."""
        if not options:
            return ""

        if len(options) > 3:
            return "[错误] options 最多提供 3 个备选方案"

        labels = set()
        for i, opt in enumerate(options):
            if not isinstance(opt, dict):
                return f"[错误] options[{i}] 必须是字典"
            label = opt.get("label", "")
            if not label or not isinstance(label, str):
                return f"[错误] options[{i}] 缺少 label 或类型错误"
            if len(label) > 80:
                return f"[错误] options[{i}] label 超过 80 字符: {label[:80]}..."
            if label.lower() in _RESERVED_LABELS:
                return f"[错误] options[{i}] label 使用了保留词: {label!r}"
            if label in labels:
                return f"[错误] options label 重复: {label!r}"
            labels.add(label)

            desc = opt.get("description", "")
            if not isinstance(desc, str):
                return f"[错误] options[{i}] description 必须是字符串"

        return ""
