"""后台任务管理工具测试."""

import time

import pytest

from ai_coding.tools.shell_tools import ExecuteCommandTool
from ai_coding.tools.task_tools import TaskListTool, TaskOutputTool, TaskStopTool


def _extract_task_id(result: str) -> str:
    """从 ExecuteCommandTool 后台任务返回结果中提取 task_id."""
    for line in result.split("\n"):
        if line.startswith("[后台任务已启动] "):
            return line.split("]", 1)[1].strip()
    raise ValueError(f"无法从结果中提取 task_id: {result!r}")


@pytest.fixture
def task_manager(isolated_work_dir):
    """创建并配置 ExecuteCommandTool 作为任务管理器."""
    tool = ExecuteCommandTool()
    tool.set_work_dir(str(isolated_work_dir))
    return tool


class TestTaskListTool:
    """task_list 工具测试."""

    def test_empty_list(self, task_manager):
        """无任务时返回提示."""
        tool = TaskListTool(task_manager)
        result = tool.execute()
        assert "没有" in result

    def test_list_running_tasks(self, task_manager):
        """列出运行中的后台任务."""
        task_manager.execute(command="sleep 10", description="long task", run_in_background=True)
        tool = TaskListTool(task_manager)
        result = tool.execute(active_only=True)
        assert "后台任务" in result
        assert "long task" in result

    def test_list_all_tasks(self, task_manager):
        """列出所有任务（包括已终止）."""
        tid = _extract_task_id(task_manager.execute(command="echo done", description="quick task", run_in_background=True))
        # 等待完成
        for _ in range(50):
            t = task_manager.get_task(tid)
            if t["status"] != "running":
                break
            time.sleep(0.05)

        tool = TaskListTool(task_manager)
        result = tool.execute(active_only=False)
        assert "quick task" in result

    def test_list_limit(self, task_manager):
        """limit 参数限制返回数量."""
        for i in range(3):
            _extract_task_id(task_manager.execute(command=f"sleep {10 + i}", description=f"task {i}", run_in_background=True))

        tool = TaskListTool(task_manager)
        result = tool.execute(active_only=True, limit=2)
        lines = [l for l in result.split("\n") if l.strip().startswith("[")]
        # 头部一行 + 最多 2 个任务
        assert len(lines) <= 3


class TestTaskOutputTool:
    """task_output 工具测试."""

    def test_output_nonexistent_task(self, task_manager):
        """查看不存在的任务应返回错误."""
        tool = TaskOutputTool(task_manager)
        result = tool.execute(task_id="not_exist")
        assert "错误" in result
        assert "不存在" in result

    def test_output_running_task(self, task_manager):
        """查看运行中的任务应返回提示."""
        tid = _extract_task_id(task_manager.execute(command="sleep 10", description="running", run_in_background=True))
        tool = TaskOutputTool(task_manager)
        result = tool.execute(task_id=tid)
        assert "running" in result or "运行中" in result

    def test_output_completed_task(self, task_manager):
        """查看已完成的任务应包含输出."""
        tid = _extract_task_id(task_manager.execute(command="echo hello", description="echo", run_in_background=True))
        # 等待完成
        for _ in range(50):
            t = task_manager.get_task(tid)
            if t["status"] != "running":
                break
            time.sleep(0.05)

        tool = TaskOutputTool(task_manager)
        result = tool.execute(task_id=tid)
        assert "hello" in result
        assert "completed" in result or "Status" in result

    def test_output_block_waits_for_completion(self, task_manager):
        """block=true 时应等待任务完成."""
        tid = _extract_task_id(task_manager.execute(command="echo delayed", description="delayed", run_in_background=True))
        tool = TaskOutputTool(task_manager)
        result = tool.execute(task_id=tid, block=True, timeout=5)
        assert "delayed" in result

    def test_output_invalid_timeout(self, task_manager):
        """timeout 越界应被钳制."""
        tid = _extract_task_id(task_manager.execute(command="echo hello", description="echo", run_in_background=True))
        tool = TaskOutputTool(task_manager)
        # 不会报错，timeout 被 clamp 到 [0, 3600]
        result = tool.execute(task_id=tid, timeout=-1)
        assert "Status" in result


class TestTaskStopTool:
    """task_stop 工具测试."""

    def test_stop_running_task(self, task_manager):
        """停止运行中的任务."""
        tid = _extract_task_id(task_manager.execute(command="sleep 30", description="stoppable", run_in_background=True))
        tool = TaskStopTool(task_manager)
        result = tool.execute(task_id=tid, reason="test stop")
        assert "Stopped" in result or "停止" in result
        t = task_manager.get_task(tid)
        assert t["status"] == "stopped"

    def test_stop_nonexistent_task(self, task_manager):
        """停止不存在的任务应返回错误."""
        tool = TaskStopTool(task_manager)
        result = tool.execute(task_id="not_exist")
        assert "错误" in result
        assert "不存在" in result

    def test_stop_already_terminated_task(self, task_manager):
        """停止已终止的任务应返回提示."""
        tid = _extract_task_id(task_manager.execute(command="echo done", description="done", run_in_background=True))
        # 等待完成
        for _ in range(50):
            t = task_manager.get_task(tid)
            if t["status"] != "running":
                break
            time.sleep(0.05)

        tool = TaskStopTool(task_manager)
        result = tool.execute(task_id=tid)
        assert "已处于终止状态" in result
