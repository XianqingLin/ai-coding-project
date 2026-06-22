"""Git 工具单元测试."""

import subprocess

import pytest

from ai_coding.tools.git_tools import (
    GitAddTool,
    GitBranchCreateTool,
    GitBranchListTool,
    GitBranchSwitchTool,
    GitCommitTool,
    GitDiffTool,
    GitLogTool,
    GitPushTool,
    GitStatusTool,
)


@pytest.fixture
def git_status_tool(isolated_work_dir):
    tool = GitStatusTool()
    tool.set_work_dir(str(isolated_work_dir))
    return tool


@pytest.fixture
def git_diff_tool(isolated_work_dir):
    tool = GitDiffTool()
    tool.set_work_dir(str(isolated_work_dir))
    return tool


@pytest.fixture
def git_log_tool(isolated_work_dir):
    tool = GitLogTool()
    tool.set_work_dir(str(isolated_work_dir))
    return tool


@pytest.fixture
def git_branch_list_tool(isolated_work_dir):
    tool = GitBranchListTool()
    tool.set_work_dir(str(isolated_work_dir))
    return tool


@pytest.fixture
def git_branch_create_tool(isolated_work_dir):
    tool = GitBranchCreateTool()
    tool.set_work_dir(str(isolated_work_dir))
    return tool


@pytest.fixture
def git_branch_switch_tool(isolated_work_dir):
    tool = GitBranchSwitchTool()
    tool.set_work_dir(str(isolated_work_dir))
    return tool


@pytest.fixture
def git_add_tool(isolated_work_dir):
    tool = GitAddTool()
    tool.set_work_dir(str(isolated_work_dir))
    return tool


@pytest.fixture
def git_commit_tool(isolated_work_dir):
    tool = GitCommitTool()
    tool.set_work_dir(str(isolated_work_dir))
    return tool


@pytest.fixture
def git_push_tool(isolated_work_dir):
    tool = GitPushTool()
    tool.set_work_dir(str(isolated_work_dir))
    return tool


def init_git_repo(path):
    """初始化一个带用户配置的 git 仓库."""
    subprocess.run(
        ["git", "init"], cwd=str(path), check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=str(path),
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(path),
        check=True,
        capture_output=True,
        text=True,
    )


def make_commit(path, filename, content, message):
    """创建文件并提交."""
    file_path = path / filename
    file_path.write_text(content, encoding="utf-8")
    subprocess.run(
        ["git", "add", filename],
        cwd=str(path),
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=str(path),
        check=True,
        capture_output=True,
        text=True,
        input="\n",
    )


class TestGitStatusTool:
    def test_status_clean_repo(self, git_status_tool, isolated_work_dir):
        init_git_repo(isolated_work_dir)
        result = git_status_tool.execute()
        assert "[成功]" in result.data

    def test_status_modified_file(self, git_status_tool, isolated_work_dir):
        init_git_repo(isolated_work_dir)
        (isolated_work_dir / "a.txt").write_text("changed", encoding="utf-8")
        subprocess.run(
            ["git", "add", "a.txt"],
            cwd=str(isolated_work_dir),
            check=True,
            capture_output=True,
        )

        result = git_status_tool.execute()

        assert "A  a.txt" in result.data

    def test_status_not_a_repo(self, git_status_tool):
        result = git_status_tool.execute()
        assert result.data.startswith("[错误]")
        assert "不是 git 仓库" in result.data


class TestGitDiffTool:
    def test_diff_working_dir(self, git_diff_tool, isolated_work_dir):
        init_git_repo(isolated_work_dir)
        make_commit(isolated_work_dir, "a.txt", "original", "init")
        (isolated_work_dir / "a.txt").write_text("modified", encoding="utf-8")

        result = git_diff_tool.execute()

        assert "modified" in result.data
        assert "original" in result.data

    def test_diff_cached(self, git_diff_tool, isolated_work_dir):
        init_git_repo(isolated_work_dir)
        make_commit(isolated_work_dir, "a.txt", "original", "init")
        (isolated_work_dir / "a.txt").write_text("modified", encoding="utf-8")
        subprocess.run(
            ["git", "add", "a.txt"],
            cwd=str(isolated_work_dir),
            check=True,
            capture_output=True,
        )

        result = git_diff_tool.execute(cached=True)

        assert "modified" in result.data

    def test_diff_not_a_repo(self, git_diff_tool):
        result = git_diff_tool.execute()
        assert result.data.startswith("[错误]")


class TestGitLogTool:
    def test_log_oneline(self, git_log_tool, isolated_work_dir):
        init_git_repo(isolated_work_dir)
        make_commit(isolated_work_dir, "a.txt", "1", "first")
        make_commit(isolated_work_dir, "a.txt", "2", "second")

        result = git_log_tool.execute(limit=10)

        assert "first" in result.data
        assert "second" in result.data

    def test_log_path_filter(self, git_log_tool, isolated_work_dir):
        init_git_repo(isolated_work_dir)
        make_commit(isolated_work_dir, "a.txt", "1", "commit a")
        make_commit(isolated_work_dir, "b.txt", "1", "commit b")

        result = git_log_tool.execute(path="a.txt")

        assert "commit a" in result.data
        assert "commit b" not in result.data

    def test_log_not_a_repo(self, git_log_tool):
        result = git_log_tool.execute()
        assert result.data.startswith("[错误]")


class TestGitBranchTools:
    def test_branch_list(self, git_branch_list_tool, isolated_work_dir):
        init_git_repo(isolated_work_dir)
        make_commit(isolated_work_dir, "a.txt", "1", "init")

        result = git_branch_list_tool.execute()

        assert "master" in result.data or "main" in result.data

    def test_branch_create(self, git_branch_create_tool, isolated_work_dir):
        init_git_repo(isolated_work_dir)
        make_commit(isolated_work_dir, "a.txt", "1", "init")

        result = git_branch_create_tool.execute(branch="feature")

        assert result.data.startswith("[成功]") or "feature" in result.data
        branches = subprocess.run(
            ["git", "branch", "--list"],
            cwd=str(isolated_work_dir),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert "feature" in branches

    def test_branch_create_requires_approval(self):
        assert GitBranchCreateTool.requires_approval is True

    def test_branch_switch(self, git_branch_switch_tool, isolated_work_dir):
        init_git_repo(isolated_work_dir)
        make_commit(isolated_work_dir, "a.txt", "1", "init")
        subprocess.run(
            ["git", "branch", "feature"],
            cwd=str(isolated_work_dir),
            check=True,
            capture_output=True,
        )

        result = git_branch_switch_tool.execute(branch="feature")

        assert result.data.startswith("[成功]") or "feature" in result.data
        current = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(isolated_work_dir),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert current == "feature"

    def test_branch_switch_create(self, git_branch_switch_tool, isolated_work_dir):
        init_git_repo(isolated_work_dir)
        make_commit(isolated_work_dir, "a.txt", "1", "init")

        result = git_branch_switch_tool.execute(branch="feature", create=True)

        assert result.data.startswith("[成功]") or "feature" in result.data
        current = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(isolated_work_dir),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert current == "feature"

    def test_branch_switch_requires_approval(self):
        assert GitBranchSwitchTool.requires_approval is True


class TestGitAddTool:
    def test_add_single_file(self, git_add_tool, isolated_work_dir):
        init_git_repo(isolated_work_dir)
        (isolated_work_dir / "a.txt").write_text("hello", encoding="utf-8")

        result = git_add_tool.execute(paths="a.txt")

        assert result.data.startswith("[成功]")
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=str(isolated_work_dir),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert "A  a.txt" in status

    def test_add_multiple_paths(self, git_add_tool, isolated_work_dir):
        init_git_repo(isolated_work_dir)
        (isolated_work_dir / "a.txt").write_text("a", encoding="utf-8")
        (isolated_work_dir / "b.txt").write_text("b", encoding="utf-8")

        result = git_add_tool.execute(paths="a.txt, b.txt")

        assert result.data.startswith("[成功]")

    def test_add_requires_approval(self):
        assert GitAddTool.requires_approval is True


class TestGitCommitTool:
    def test_commit(self, git_commit_tool, git_add_tool, isolated_work_dir):
        init_git_repo(isolated_work_dir)
        (isolated_work_dir / "a.txt").write_text("hello", encoding="utf-8")
        git_add_tool.execute(paths="a.txt")

        result = git_commit_tool.execute(message="add a.txt")

        assert result.data.startswith("[成功]") or "add a.txt" in result.data
        log = subprocess.run(
            ["git", "log", "--oneline", "-n", "1"],
            cwd=str(isolated_work_dir),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert "add a.txt" in log

    def test_commit_requires_approval(self):
        assert GitCommitTool.requires_approval is True

    def test_commit_empty_message(self, git_commit_tool):
        result = git_commit_tool.execute(message="   ")
        assert result.data.startswith("[错误]")


class TestGitPushTool:
    def test_push_no_remote(self, git_push_tool, isolated_work_dir):
        init_git_repo(isolated_work_dir)
        make_commit(isolated_work_dir, "a.txt", "1", "init")

        result = git_push_tool.execute()

        # 没有远程仓库, push 应该失败并被包装为错误
        assert result.data.startswith("[错误]")

    def test_push_requires_approval(self):
        assert GitPushTool.requires_approval is True


class TestGitSandboxAndErrors:
    def test_status_outside_work_dir(self, git_status_tool):
        result = git_status_tool.execute(cwd="../outside")
        assert result.data.startswith("[错误]")

    def test_branch_create_empty_name(self, git_branch_create_tool):
        result = git_branch_create_tool.execute(branch="   ")
        assert result.data.startswith("[错误]")

    def test_add_empty_paths(self, git_add_tool):
        result = git_add_tool.execute(paths="   ")
        assert result.data.startswith("[错误]")


class TestGitOutputTruncation:
    def test_truncate_by_lines(self):
        from ai_coding.tools.git_tools import _truncate_output

        text = "\n".join(f"line {i}" for i in range(250))
        result = _truncate_output(text)
        assert "已截断" in result
        assert result.count("\n") <= 205  # 200 行 + 截断提示

    def test_truncate_by_bytes(self):
        from ai_coding.tools.git_tools import _truncate_output

        # 构造超过 32KB 的文本，但行数不超过 200
        text = ("x" * 500 + "\n") * 100  # 约 50KB
        result = _truncate_output(text)
        assert "已截断" in result
        assert len(result.encode("utf-8")) <= 33000
