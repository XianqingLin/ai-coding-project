"""文件类工具单元测试."""

import pytest

from ai_coding.tools.file_tools import (
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
    SearchReplaceTool,
    WriteFileTool,
)


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


@pytest.fixture
def sr_tool(isolated_work_dir):
    tool = SearchReplaceTool()
    tool.set_work_dir(str(isolated_work_dir))
    return tool


class TestReadFileTool:
    def test_read_existing_file(self, read_tool, isolated_work_dir):
        file_path = isolated_work_dir / "hello.txt"
        file_path.write_text("line1\nline2\nline3", encoding="utf-8")

        result = read_tool.execute("hello.txt")

        assert "文件: " in result.data
        assert "line1" in result.data
        assert "line2" in result.data
        assert "共 3 行" in result.data

    def test_read_nonexistent_file(self, read_tool):
        result = read_tool.execute("not_exist.txt")
        assert result.data.startswith("[错误]")

    def test_read_directory(self, read_tool, isolated_work_dir):
        (isolated_work_dir / "adir").mkdir()
        result = read_tool.execute("adir")
        assert "是一个目录" in result.data

    def test_read_with_offset_and_limit(self, read_tool, isolated_work_dir):
        file_path = isolated_work_dir / "nums.txt"
        file_path.write_text(
            "\n".join(f"line{i}" for i in range(1, 11)), encoding="utf-8"
        )

        result = read_tool.execute("nums.txt", line_offset=3, n_lines=4)

        assert "3 | line3" in result.data
        assert "6 | line6" in result.data
        assert "line2" not in result.data
        assert "line7" not in result.data

    def test_read_long_line_truncation(self, read_tool, isolated_work_dir):
        file_path = isolated_work_dir / "long.txt"
        file_path.write_text("x" * 3000, encoding="utf-8")

        result = read_tool.execute("long.txt")
        assert "..." in result.data
        assert len(result.data) < 3500

    def test_read_over_1000_lines(self, read_tool, isolated_work_dir):
        file_path = isolated_work_dir / "many.txt"
        file_path.write_text(
            "\n".join(f"line{i}" for i in range(1500)), encoding="utf-8"
        )

        result = read_tool.execute("many.txt")
        assert "超过 1000 行" in result.data
        assert "line_offset=" in result.data

    def test_read_binary_file(self, read_tool, isolated_work_dir):
        file_path = isolated_work_dir / "binary.bin"
        file_path.write_bytes(b"\x00\x01\x02\xff")

        result = read_tool.execute("binary.bin")
        assert "二进制文件" in result.data


class TestWriteFileTool:
    def test_write_new_file(self, write_tool, isolated_work_dir):
        result = write_tool.execute("new.txt", "hello world")

        assert result.data.startswith("[成功]")
        assert (isolated_work_dir / "new.txt").read_text(
            encoding="utf-8"
        ) == "hello world"

    def test_write_nested_file(self, write_tool, isolated_work_dir):
        result = write_tool.execute("a/b/c.txt", "nested")

        assert result.data.startswith("[成功]")
        assert (isolated_work_dir / "a" / "b" / "c.txt").read_text(
            encoding="utf-8"
        ) == "nested"


class TestListDirTool:
    def test_list_directory(self, list_dir_tool, isolated_work_dir):
        (isolated_work_dir / "file1.txt").write_text("a", encoding="utf-8")
        (isolated_work_dir / "file2.py").write_text("b", encoding="utf-8")

        result = list_dir_tool.execute(".")

        assert "file1.txt" in result.data
        assert "file2.py" in result.data


class TestGrepTool:
    def test_grep_pattern(self, grep_tool, isolated_work_dir):
        (isolated_work_dir / "a.py").write_text(
            "def foo():\n    pass\n", encoding="utf-8"
        )
        (isolated_work_dir / "b.py").write_text(
            "def bar():\n    pass\n", encoding="utf-8"
        )

        result = grep_tool.execute(pattern="def foo", path=".")

        assert "a.py" in result.data
        assert "b.py" not in result.data


class TestGlobTool:
    def test_glob_pattern(self, glob_tool, isolated_work_dir):
        (isolated_work_dir / "a.py").write_text("a", encoding="utf-8")
        (isolated_work_dir / "b.py").write_text("b", encoding="utf-8")
        (isolated_work_dir / "c.txt").write_text("c", encoding="utf-8")

        result = glob_tool.execute("*.py")

        assert "a.py" in result.data
        assert "b.py" in result.data
        assert "c.txt" not in result.data


class TestSearchReplaceTool:
    @staticmethod
    def _block(search: str, replace: str) -> str:
        return (
            f"<<<<<<< SEARCH\n"
            f"{search}\n"
            f"=======\n"
            f"{replace}\n"
            f">>>>>>> REPLACE\n"
        )

    def test_single_block_success(self, sr_tool, isolated_work_dir):
        file_path = isolated_work_dir / "src.py"
        file_path.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

        blocks = self._block(
            "def add(a, b):\n    return a + b", "def add(a, b):\n    return a - b"
        )
        result = sr_tool.execute("src.py", blocks)

        assert result.data.startswith("[成功]")
        assert "应用 1 个编辑块" in result.data
        assert (
            file_path.read_text(encoding="utf-8")
            == "def add(a, b):\n    return a - b\n"
        )

    def test_multiple_blocks_success(self, sr_tool, isolated_work_dir):
        file_path = isolated_work_dir / "src.py"
        file_path.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")

        blocks = self._block("line1", "line1 edited") + self._block(
            "line3", "line3 edited"
        )
        result = sr_tool.execute("src.py", blocks)

        assert result.data.startswith("[成功]")
        assert "应用 2 个编辑块" in result.data
        assert (
            file_path.read_text(encoding="utf-8")
            == "line1 edited\nline2\nline3 edited\nline4\n"
        )

    def test_no_match_returns_error(self, sr_tool, isolated_work_dir):
        file_path = isolated_work_dir / "src.py"
        file_path.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

        blocks = self._block("not in file", "replacement")
        result = sr_tool.execute("src.py", blocks)

        assert result.data.startswith("[错误]")
        assert "未找到精确匹配内容" in result.data
        assert (
            file_path.read_text(encoding="utf-8")
            == "def add(a, b):\n    return a + b\n"
        )

    def test_fuzzy_match_suggestion(self, sr_tool, isolated_work_dir):
        file_path = isolated_work_dir / "src.py"
        file_path.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

        blocks = self._block("    return a + c", "replacement")
        result = sr_tool.execute("src.py", blocks)

        assert result.data.startswith("[错误]")
        assert "最接近的匹配" in result.data

    def test_multiple_occurrences_returns_error(self, sr_tool, isolated_work_dir):
        file_path = isolated_work_dir / "src.py"
        file_path.write_text("foo\nfoo\nfoo\n", encoding="utf-8")

        blocks = self._block("foo", "bar")
        result = sr_tool.execute("src.py", blocks)

        assert result.data.startswith("[错误]")
        assert "匹配内容不唯一" in result.data

    def test_missing_divider_returns_error(self, sr_tool, isolated_work_dir):
        file_path = isolated_work_dir / "src.py"
        file_path.write_text("hello\n", encoding="utf-8")

        blocks = "<<<<<<< SEARCH\nhello\n>>>>>>> REPLACE\n"
        result = sr_tool.execute("src.py", blocks)

        assert result.data.startswith("[错误]")
        assert "缺少 '======='" in result.data

    def test_missing_end_marker_returns_error(self, sr_tool, isolated_work_dir):
        file_path = isolated_work_dir / "src.py"
        file_path.write_text("hello\n", encoding="utf-8")

        blocks = "<<<<<<< SEARCH\nhello\n=======\nworld\n"
        result = sr_tool.execute("src.py", blocks)

        assert result.data.startswith("[错误]")
        assert "缺少 '>>>>>>> REPLACE'" in result.data

    def test_sandbox_violation_returns_error(self, sr_tool, isolated_work_dir):
        file_path = isolated_work_dir / "src.py"
        file_path.write_text("hello\n", encoding="utf-8")

        blocks = self._block("hello", "world")
        result = sr_tool.execute("../src.py", blocks)

        assert result.data.startswith("[错误]")
        assert file_path.read_text(encoding="utf-8") == "hello\n"

    def test_empty_search_returns_error(self, sr_tool, isolated_work_dir):
        file_path = isolated_work_dir / "src.py"
        file_path.write_text("hello\n", encoding="utf-8")

        # 空 SEARCH：<<<<<<< SEARCH 与 ======= 之间只有换行
        blocks = "<<<<<<< SEARCH\n=======\nworld\n>>>>>>> REPLACE\n"
        result = sr_tool.execute("src.py", blocks)

        assert result.data.startswith("[错误]")
        assert "SEARCH 内容不能为空" in result.data

    def test_no_blocks_returns_error(self, sr_tool, isolated_work_dir):
        file_path = isolated_work_dir / "src.py"
        file_path.write_text("hello\n", encoding="utf-8")

        result = sr_tool.execute("src.py", "not a valid block")

        assert result.data.startswith("[错误]")
        assert "未找到任何 SEARCH/REPLACE 编辑块" in result.data
