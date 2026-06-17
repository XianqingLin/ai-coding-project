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
