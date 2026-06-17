"""文件类工具单元测试."""

from pathlib import Path

import pytest

from ai_coding.tools.file_tools import EditFile, GlobTool, GrepTool, ListDirTool, ReadFileTool, WriteFileTool


@pytest.fixture
def read_tool(isolated_work_dir):
    tool = ReadFileTool()
    tool.set_work_dir(str(isolated_work_dir))
    return tool


@pytest.fixture
def write_tool(isolated_work_dir):
    tool = WriteFileTool()
    tool.set_work_dir(str(isolated_work_dir))
    return tool


@pytest.fixture
def edit_tool(isolated_work_dir):
    tool = EditFile()
    tool.set_work_dir(str(isolated_work_dir))
    return tool


@pytest.fixture
def list_dir_tool(isolated_work_dir):
    tool = ListDirTool()
    tool.set_work_dir(str(isolated_work_dir))
    return tool


@pytest.fixture
def grep_tool(isolated_work_dir):
    tool = GrepTool()
    tool.set_work_dir(str(isolated_work_dir))
    return tool


@pytest.fixture
def glob_tool(isolated_work_dir):
    tool = GlobTool()
    tool.set_work_dir(str(isolated_work_dir))
    return tool


class TestReadFileTool:
    def test_read_existing_file(self, read_tool, isolated_work_dir):
        file_path = isolated_work_dir / "hello.txt"
        file_path.write_text("line1\nline2\nline3", encoding="utf-8")

        result = read_tool.execute("hello.txt")

        assert "文件: " in result
        assert "line1" in result
        assert "line2" in result
        assert "共 3 行" in result

    def test_read_nonexistent_file(self, read_tool):
        result = read_tool.execute("not_exist.txt")
        assert result.startswith("[错误]")

    def test_read_with_offset_and_limit(self, read_tool, isolated_work_dir):
        file_path = isolated_work_dir / "nums.txt"
        file_path.write_text("\n".join(f"line{i}" for i in range(1, 11)), encoding="utf-8")

        result = read_tool.execute("nums.txt", line_offset=3, n_lines=4)

        assert "3 | line3" in result
        assert "6 | line6" in result
        assert "line2" not in result
        assert "line7" not in result


class TestWriteFileTool:
    def test_write_new_file(self, write_tool, isolated_work_dir):
        result = write_tool.execute("new.txt", "hello world")

        assert result.startswith("[成功]")
        assert (isolated_work_dir / "new.txt").read_text(encoding="utf-8") == "hello world"

    def test_write_nested_file(self, write_tool, isolated_work_dir):
        result = write_tool.execute("a/b/c.txt", "nested")

        assert result.startswith("[成功]")
        assert (isolated_work_dir / "a" / "b" / "c.txt").read_text(encoding="utf-8") == "nested"


class TestEditFileTool:
    def test_edit_exact_match(self, edit_tool, isolated_work_dir):
        file_path = isolated_work_dir / "src.py"
        file_path.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

        result = edit_tool.execute(
            "src.py",
            old_string="    return a + b",
            new_string="    return a - b",
        )

        assert result.startswith("[成功]")
        assert "return a - b" in file_path.read_text(encoding="utf-8")

    def test_edit_no_match_returns_error(self, edit_tool, isolated_work_dir):
        file_path = isolated_work_dir / "src.py"
        file_path.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

        result = edit_tool.execute(
            "src.py",
            old_string="not in file",
            new_string="replacement",
        )

        assert result.startswith("[错误]")


class TestListDirTool:
    def test_list_directory(self, list_dir_tool, isolated_work_dir):
        (isolated_work_dir / "file1.txt").write_text("a", encoding="utf-8")
        (isolated_work_dir / "file2.py").write_text("b", encoding="utf-8")

        result = list_dir_tool.execute(".")

        assert "file1.txt" in result
        assert "file2.py" in result


class TestGrepTool:
    def test_grep_pattern(self, grep_tool, isolated_work_dir):
        (isolated_work_dir / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        (isolated_work_dir / "b.py").write_text("def bar():\n    pass\n", encoding="utf-8")

        result = grep_tool.execute(pattern="def foo", path=".")

        assert "a.py" in result
        assert "b.py" not in result


class TestGlobTool:
    def test_glob_pattern(self, glob_tool, isolated_work_dir):
        (isolated_work_dir / "a.py").write_text("a", encoding="utf-8")
        (isolated_work_dir / "b.py").write_text("b", encoding="utf-8")
        (isolated_work_dir / "c.txt").write_text("c", encoding="utf-8")

        result = glob_tool.execute("*.py")

        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result
