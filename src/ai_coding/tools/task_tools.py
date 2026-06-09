"""后台任务管理工具.

提供 task_list、task_output、task_stop 三个工具，用于管理通过 execute_command
后台模式启动的异步任务.
"""

import time
from typing import List, Optional

from ai_coding.tools.base import Tool, ToolParameter
from ai_coding.tools.shell_tools import ExecuteCommandTool


class TaskListTool(Tool):
    """列出后台任务."""

    name = "task_list"
    requires_approval = False
    description = (
        "列出后台任务. 可以查看所有任务或仅查看运行中的任务.\n"
        "通过 execute_command 的 run_in_background=true 启动的任务会出现在此列表中."
    )

    def __init__(self, task_manager: ExecuteCommandTool) -> None:
        super().__init__()
        self._task_manager = task_manager

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "active_only",
                "boolean",
                "仅列出运行中的任务，默认 true",
                required=False,
                default=True,
            ),
            ToolParameter(
                "limit",
                "integer",
                "返回数量上限，默认 20，范围 1-100",
                required=False,
                default=20,
            ),
        ]

    def execute(self, active_only: bool = True, limit: int = 20) -> str:
        limit = max(1, min(100, int(limit)))
        tasks = self._task_manager.list_tasks(active_only=active_only, limit=limit)

        if not tasks:
            scope = "运行中" if active_only else "所有"
            return f"当前没有 {scope} 的后台任务."

        lines = [f"后台任务 ({len(tasks)} 个):"]
        for t in tasks:
            status = t["status"]
            desc = t["description"]
            cmd = t["command"]
            if len(cmd) > 60:
                cmd = cmd[:60] + "..."
            lines.append(f"  [{status:10s}] {t['task_id']}  {desc}  ({cmd})")

        return "\n".join(lines)


class TaskOutputTool(Tool):
    """查看后台任务的输出."""

    name = "task_output"
    requires_approval = False
    description = (
        "查看指定后台任务的状态与输出. 返回最近 32KB 的内联预览，"
        "完整日志可通过返回的 output_path 用 read_file 读取.\n"
        "block=true 时可等待任务完成后再返回."
    )

    def __init__(self, task_manager: ExecuteCommandTool) -> None:
        super().__init__()
        self._task_manager = task_manager

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("task_id", "string", "任务 ID"),
            ToolParameter(
                "block",
                "boolean",
                "是否阻塞等待任务完成后再返回，默认 false",
                required=False,
                default=False,
            ),
            ToolParameter(
                "timeout",
                "integer",
                "等待秒数，默认 30，范围 0-3600",
                required=False,
                default=30,
            ),
        ]

    def execute(self, task_id: str, block: bool = False, timeout: int = 30) -> str:
        timeout = max(0, min(3600, int(timeout)))

        if block:
            start = time.time()
            while time.time() - start < timeout:
                t = self._task_manager.get_task(task_id)
                if not t:
                    return f"[错误] 任务不存在: {task_id}"
                if t["status"] != "running":
                    break
                time.sleep(0.5)

        t = self._task_manager.get_task(task_id)
        if not t:
            return f"[错误] 任务不存在: {task_id}"

        status = t["status"]
        output_path = t["output_path"]
        preview = self._task_manager.read_task_output(task_id, max_bytes=32 * 1024)

        header = (
            f"[Task] {task_id}\n"
            f"[Status] {status}\n"
            f"[Command] {t['command']}\n"
            f"[Description] {t['description']}\n"
            f"[Output Path] {output_path}\n"
        )
        if t.get("exit_code") is not None:
            header += f"[Exit Code] {t['exit_code']}\n"

        if preview:
            header += f"\n[Output Preview (last 32KB)]\n{'-'*50}\n{preview}\n{'-'*50}"
        else:
            header += "\n[Output] 暂无输出."

        if status == "running":
            header += "\n[提示] 任务仍在运行中，可稍后再次调用 task_output 查看最新输出。"

        return header


class TaskStopTool(Tool):
    """停止后台任务."""

    name = "task_stop"
    requires_approval = True
    description = (
        "停止指定的后台任务. 对已处于终止状态的任务也能安全调用（会返回提示而不报错）.\n"
        "此操作需要用户授权，因为会强制终止进程。"
    )

    def __init__(self, task_manager: ExecuteCommandTool) -> None:
        super().__init__()
        self._task_manager = task_manager

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("task_id", "string", "要停止的任务 ID"),
            ToolParameter(
                "reason",
                "string",
                "停止原因，默认 'Stopped by TaskStop'",
                required=False,
                default="Stopped by TaskStop",
            ),
        ]

    def execute(self, task_id: str, reason: str = "Stopped by TaskStop") -> str:
        return self._task_manager.stop_task(task_id, reason)
