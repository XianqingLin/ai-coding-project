"""CLI 集成测试.

使用 Typer 的 CliRunner 和 Mock LLM 验证 CLI 命令流程.
"""

import pytest
from typer.testing import CliRunner

from ai_coding import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def mock_provider(monkeypatch):
    """强制 CLI 使用 Mock LLM，避免消耗真实 API."""
    monkeypatch.setattr("ai_coding.agent.service.DEFAULT_LLM_PROVIDER", "mock")


class TestCLISessionCommands:
    def test_session_list(self, runner, isolated_work_dir):
        result = runner.invoke(
            cli.app, ["session", "list", "--work-dir", str(isolated_work_dir)]
        )
        assert result.exit_code == 0
        # SessionManager 会自动创建一个 default 会话
        assert "default" in result.output

    def test_session_new(self, runner, isolated_work_dir):
        result = runner.invoke(
            cli.app,
            ["session", "new", "my-session", "--work-dir", str(isolated_work_dir)],
        )
        assert result.exit_code == 0
        assert "Created session:" in result.output

    def test_session_switch_and_delete(self, runner, isolated_work_dir):
        # 先创建两个会话
        runner.invoke(
            cli.app,
            ["session", "new", "session-a", "--work-dir", str(isolated_work_dir)],
        )
        result = runner.invoke(
            cli.app, ["session", "list", "--work-dir", str(isolated_work_dir)]
        )
        sessions = result.output.strip().split("\n")
        # 找到 session-a 的 ID
        sid_a = None
        for line in sessions:
            if "session-a" in line:
                sid_a = line.split()[1]
                break
        assert sid_a

        # 切换
        result = runner.invoke(
            cli.app, ["session", "switch", sid_a, "--work-dir", str(isolated_work_dir)]
        )
        assert result.exit_code == 0
        assert f"Switched to: {sid_a}" in result.output

        # 删除
        result = runner.invoke(
            cli.app, ["session", "delete", sid_a, "--work-dir", str(isolated_work_dir)]
        )
        assert result.exit_code == 0
        assert f"Deleted session: {sid_a}" in result.output

    def test_session_switch_not_found(self, runner, isolated_work_dir):
        result = runner.invoke(
            cli.app,
            ["session", "switch", "not-exist", "--work-dir", str(isolated_work_dir)],
        )
        assert result.exit_code == 1
        assert "Session not found" in result.output


class TestCLIAskCommand:
    def test_ask(self, runner, isolated_work_dir):
        result = runner.invoke(
            cli.app,
            ["ask", "hello", "--work-dir", str(isolated_work_dir), "--auto-approve"],
        )
        assert result.exit_code == 0
        # Mock LLM 默认回复
        assert "MockLLM" in result.output or "hello" in result.output.lower()

    def test_ask_with_session(self, runner, isolated_work_dir):
        # 创建会话
        result = runner.invoke(
            cli.app,
            ["session", "new", "test-sess", "--work-dir", str(isolated_work_dir)],
        )
        sessions = runner.invoke(
            cli.app, ["session", "list", "--work-dir", str(isolated_work_dir)]
        ).output
        sid = None
        for line in sessions.strip().split("\n"):
            if "test-sess" in line:
                sid = line.split()[1]
                break
        assert sid

        # 使用指定会话 ask
        result = runner.invoke(
            cli.app,
            [
                "ask",
                "hi",
                "--work-dir",
                str(isolated_work_dir),
                "--auto-approve",
                "--session",
                sid,
            ],
        )
        assert result.exit_code == 0
