"""AGENTS.md 生成器单元测试."""

from pathlib import Path

from ai_coding.agents_md_generator import (
    _collect_project_info,
    _list_directory_tree,
    generate_agents_md,
    write_agents_md,
)
from ai_coding.mock_llm import MockChatModel, mock_text


class TestCollectProjectInfo:
    def test_collects_top_level_and_readme(self, isolated_work_dir: Path) -> None:
        (isolated_work_dir / "README.md").write_text(
            "# Test Project\n\nA test project.", encoding="utf-8"
        )
        (isolated_work_dir / "src").mkdir()
        (isolated_work_dir / "tests").mkdir()

        info = _collect_project_info(isolated_work_dir)

        assert "README.md" in info["top_level"]
        assert "src" in info["top_level"]
        assert "tests" in info["top_level"]
        assert "Test Project" in info["readme"]

    def test_collects_pyproject_toml(self, isolated_work_dir: Path) -> None:
        (isolated_work_dir / "pyproject.toml").write_text(
            '[project]\nname = "demo"\n', encoding="utf-8"
        )

        info = _collect_project_info(isolated_work_dir)

        assert "pyproject.toml" in info["config_files"]
        assert 'name = "demo"' in info["config_files"]["pyproject.toml"]

    def test_collects_existing_agents_md(self, isolated_work_dir: Path) -> None:
        existing = "# Existing\n\nSome rules."
        (isolated_work_dir / "AGENTS.md").write_text(existing, encoding="utf-8")

        info = _collect_project_info(isolated_work_dir)

        assert info["existing_agents_md"] == existing


class TestListDirectoryTree:
    def test_tree_depth_limit(self, isolated_work_dir: Path) -> None:
        (isolated_work_dir / "a" / "b" / "c").mkdir(parents=True)
        (isolated_work_dir / "a" / "file.txt").write_text("x", encoding="utf-8")

        tree = _list_directory_tree(isolated_work_dir / "a", max_depth=0)

        assert any("b" in entry for entry in tree)
        assert not any("c" in entry for entry in tree)


class TestGenerateAgentsMd:
    def test_generates_content(self, isolated_work_dir: Path) -> None:
        (isolated_work_dir / "README.md").write_text(
            "# Demo\n\nA demo project.", encoding="utf-8"
        )
        llm = MockChatModel(responses=[mock_text("# Generated AGENTS.md")])

        content = generate_agents_md(isolated_work_dir, llm=llm)

        assert content == "# Generated AGENTS.md"

    def test_prompt_includes_project_info(self, isolated_work_dir: Path) -> None:
        (isolated_work_dir / "README.md").write_text(
            "# UniqueProjectName\n\nSpecial project.", encoding="utf-8"
        )

        captured_messages: list = []
        original_invoke = MockChatModel.invoke

        def capturing_invoke(self, messages, **kwargs):
            captured_messages.extend(messages)
            return original_invoke(self, messages, **kwargs)

        llm = MockChatModel(responses=[mock_text("ok")])
        llm.invoke = lambda messages, **kwargs: capturing_invoke(
            llm, messages, **kwargs
        )

        generate_agents_md(isolated_work_dir, llm=llm)

        prompt_text = "\n".join(m.content for m in captured_messages)
        assert "UniqueProjectName" in prompt_text


class TestWriteAgentsMd:
    def test_writes_new_file(self, isolated_work_dir: Path) -> None:
        written = write_agents_md(isolated_work_dir, "# New AGENTS.md", overwrite=False)

        assert written is True
        assert (isolated_work_dir / "AGENTS.md").read_text(
            encoding="utf-8"
        ) == "# New AGENTS.md"

    def test_refuses_overwrite_without_flag(self, isolated_work_dir: Path) -> None:
        (isolated_work_dir / "AGENTS.md").write_text("# Old", encoding="utf-8")

        written = write_agents_md(isolated_work_dir, "# New", overwrite=False)

        assert written is False
        assert (isolated_work_dir / "AGENTS.md").read_text(encoding="utf-8") == "# Old"

    def test_overwrites_with_flag(self, isolated_work_dir: Path) -> None:
        (isolated_work_dir / "AGENTS.md").write_text("# Old", encoding="utf-8")

        written = write_agents_md(isolated_work_dir, "# New", overwrite=True)

        assert written is True
        assert (isolated_work_dir / "AGENTS.md").read_text(encoding="utf-8") == "# New"
