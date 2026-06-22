"""路径安全校验工具测试."""

import os
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from ai_coding.tools.safety import (
    PathBoundaryError,
    resolve_workdir_cwd,
    resolve_workdir_path,
)


class TestResolveWorkdirPath:
    def test_resolves_relative_path(self, isolated_work_dir: Path) -> None:
        result = resolve_workdir_path("foo.txt", str(isolated_work_dir))
        assert result.resolve() == (isolated_work_dir / "foo.txt").resolve()

    def test_resolves_absolute_path_within_work_dir(
        self, isolated_work_dir: Path
    ) -> None:
        abs_path = str(isolated_work_dir / "foo.txt")
        result = resolve_workdir_path(abs_path, str(isolated_work_dir))
        assert result.resolve() == (isolated_work_dir / "foo.txt").resolve()

    def test_rejects_absolute_path_when_disabled(self, isolated_work_dir: Path) -> None:
        abs_path = str(isolated_work_dir / "foo.txt")
        with pytest.raises(PathBoundaryError, match="不允许使用绝对路径"):
            resolve_workdir_path(
                abs_path, str(isolated_work_dir), allow_abs_within_work_dir=False
            )

    def test_rejects_parent_traversal(self, isolated_work_dir: Path) -> None:
        with pytest.raises(PathBoundaryError, match="路径包含 '..' 遍历"):
            resolve_workdir_path("../foo.txt", str(isolated_work_dir))

    def test_rejects_outside_absolute_path(self, isolated_work_dir: Path) -> None:
        outside = str(Path(isolated_work_dir).parent / "outside.txt")
        with pytest.raises(PathBoundaryError, match="超出工作目录"):
            resolve_workdir_path(outside, str(isolated_work_dir))

    def test_rejects_empty_path(self) -> None:
        with pytest.raises(ValueError, match="路径不能为空"):
            resolve_workdir_path("", "/tmp")

    def test_must_exist(self, isolated_work_dir: Path) -> None:
        (isolated_work_dir / "exists.txt").write_text("x", encoding="utf-8")
        result = resolve_workdir_path(
            "exists.txt", str(isolated_work_dir), must_exist=True
        )
        assert result.exists()

        with pytest.raises(ValueError, match="路径不存在"):
            resolve_workdir_path("missing.txt", str(isolated_work_dir), must_exist=True)

    def test_invalid_work_dir(self) -> None:
        with pytest.raises(ValueError, match="工作目录无效"):
            resolve_workdir_path("foo.txt", "/nonexistent/path/12345")

    def test_empty_work_dir_uses_cwd(
        self, isolated_work_dir: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(os, "getcwd", lambda: str(isolated_work_dir))
        result = resolve_workdir_path("foo.txt", "")
        assert result.resolve() == (isolated_work_dir / "foo.txt").resolve()


class TestResolveWorkdirCwd:
    def test_empty_cwd_returns_work_dir(self, isolated_work_dir: Path) -> None:
        result = resolve_workdir_cwd("", str(isolated_work_dir))
        assert result == isolated_work_dir.resolve()

    def test_cwd_must_exist(self, isolated_work_dir: Path) -> None:
        subdir = isolated_work_dir / "subdir"
        subdir.mkdir()
        result = resolve_workdir_cwd("subdir", str(isolated_work_dir))
        assert result == subdir.resolve()

    def test_cwd_outside_work_dir_rejected(self, isolated_work_dir: Path) -> None:
        with pytest.raises(PathBoundaryError):
            resolve_workdir_cwd("../", str(isolated_work_dir))
