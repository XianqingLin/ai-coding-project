"""Todo 工具单元测试."""

import pytest

from ai_coding.tools.todo_tool import TodoTool


@pytest.fixture
def todo_tool():
    return TodoTool()


class TestTodoTool:
    def test_add_and_list(self, todo_tool):
        todo_tool.execute("add", task="task A")
        todo_tool.execute("add", task="task B")

        result = todo_tool.execute("list")

        assert "task A" in result
        assert "task B" in result
        assert "进度: 0/2 已完成" in result

    def test_complete_by_index(self, todo_tool):
        todo_tool.execute("add", task="task A")
        todo_tool.execute("complete", index=1)

        result = todo_tool.execute("list")
        assert "[x]" in result
        assert "进度: 1/1 已完成" in result

    def test_complete_by_text(self, todo_tool):
        todo_tool.execute("add", task="buy milk")
        result = todo_tool.execute("complete", task="milk")

        assert "[完成]" in result

    def test_remove_by_index(self, todo_tool):
        todo_tool.execute("add", task="task A")
        todo_tool.execute("add", task="task B")
        result = todo_tool.execute("remove", index=1)

        assert "[删除] task A" in result
        assert "task A" not in todo_tool.execute("list")

    def test_update(self, todo_tool):
        todo_tool.execute("add", task="old text")
        result = todo_tool.execute("update", index=1, task="new text")

        assert "new text" in result
        assert "old text → new text" in result

    def test_add_without_task_returns_error(self, todo_tool):
        result = todo_tool.execute("add")
        assert result.startswith("[错误]")


class TestTodoToolEdgeCases:
    def test_list_empty(self, todo_tool):
        assert todo_tool.execute("list") == "当前无待办任务。"

    def test_complete_out_of_range(self, todo_tool):
        todo_tool.execute("add", task="task")
        result = todo_tool.execute("complete", index=5)
        assert "超出范围" in result

    def test_complete_no_params(self, todo_tool):
        result = todo_tool.execute("complete")
        assert "需要提供 index 或 task" in result

    def test_complete_by_text_not_found(self, todo_tool):
        todo_tool.execute("add", task="task")
        result = todo_tool.execute("complete", task="notfound")
        assert "未找到匹配的任务" in result

    def test_remove_out_of_range(self, todo_tool):
        todo_tool.execute("add", task="task")
        result = todo_tool.execute("remove", index=5)
        assert "超出范围" in result

    def test_remove_no_params(self, todo_tool):
        result = todo_tool.execute("remove")
        assert "需要提供 index 或 task" in result

    def test_remove_by_text_not_found(self, todo_tool):
        todo_tool.execute("add", task="task")
        result = todo_tool.execute("remove", task="notfound")
        assert "未找到匹配的任务" in result

    def test_update_no_task(self, todo_tool):
        todo_tool.execute("add", task="task")
        result = todo_tool.execute("update", index=1)
        assert "必须提供 task 参数" in result

    def test_update_no_index(self, todo_tool):
        todo_tool.execute("add", task="task")
        result = todo_tool.execute("update", task="new")
        assert "必须提供 index 参数" in result

    def test_update_out_of_range(self, todo_tool):
        todo_tool.execute("add", task="task")
        result = todo_tool.execute("update", index=5, task="new")
        assert "超出范围" in result

    def test_sync_and_todos_property(self, todo_tool):
        todo_tool.sync([{"task": "synced", "done": True}])
        assert todo_tool.todos == [{"task": "synced", "done": True}]
