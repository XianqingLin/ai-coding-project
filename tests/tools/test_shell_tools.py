"""Shell 与任务管理工具单元测试."""

import time

from ai_coding.tools.task_tools import TaskListTool, TaskOutputTool, TaskStopTool


class TestExecuteCommandTool:
    def test_foreground_echo(self, task_manager):
        result = task_manager.execute("echo hello")
        assert "hello" in result

    def test_foreground_stderr(self, task_manager):
        result = task_manager.execute("echo error >&2")
        assert "error" in result

    def test_foreground_timeout(self, task_manager):
        result = task_manager.execute("sleep 10", timeout=500)
        assert result.startswith("[超时]")

    def test_sandbox_violation(self, task_manager):
        result = task_manager.execute("echo hi", cwd="..")
        assert result.startswith("[错误]")


class TestBackgroundTasks:
    def test_background_start_list_stop(self, task_manager):
        list_tool = TaskListTool(task_manager=task_manager)
        output_tool = TaskOutputTool(task_manager=task_manager)
        stop_tool = TaskStopTool(task_manager=task_manager)

        start_result = task_manager.execute(
            "echo bg_done && sleep 0.5",
            run_in_background=True,
            description="test bg task",
        )
        assert "[后台任务已启动]" in start_result

        task_id = start_result.splitlines()[0].split()[-1]

        list_result = list_tool.execute()
        assert task_id in list_result

        # 等待任务完成
        time.sleep(1.0)

        output_result = output_tool.execute(task_id=task_id)
        assert "bg_done" in output_result

        stop_result = stop_tool.execute(task_id=task_id)
        # 任务可能已自然完成，也可能被主动停止；两种状态都接受
        assert (
            "[停止]" in stop_result
            or "completed" in stop_result
            or "已处于终止状态" in stop_result
        )
