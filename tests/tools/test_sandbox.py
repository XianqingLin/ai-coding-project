"""路径沙箱工具测试."""

import os
from pathlib import Path

import pytest

from ai_coding.tools.sandbox import (
    SandboxViolationError,
    resolve_sandboxed_cwd,
    resolve_sandboxed_path,
)


class TestResolveSandboxedPath:
    def test_resolves_relative_path(self, isolated_work_dir):
        result = resolve_sandboxed_path("foo.txt", str(isolated_work_dir))
        assert result == isolated_work_dir / "foo.txt"

    def test_resolves_absolute_path_within_work_dir(self, isolated_work_dir):
        abs_path = str(isolated_work_dir / "foo.txt")
        result = resolve_sandboxed_path(abs_path, str(isolated_work_dir))
        assert result == isolated_work_dir / "foo.txt"

    def test_rejects_absolute_path_when_disabled(self, isolated_work_dir):
        abs_path = str(isolated_work_dir / "foo.txt")
        with pytest.raises(SandboxViolationError, match="不允许使用绝对路径"):
            resolve_sandboxed_path(
                abs_path, str(isolated_work_dir), allow_abs_within_work_dir=False
            )

    def test_rejects_parent_traversal(self, isolated_work_dir):
        with pytest.raises(SandboxViolationError, match="路径包含 '..' 遍历"):
            resolve_sandboxed_path("../foo.txt", str(isolated_work_dir))

    def test_rejects_outside_absolute_path(self, isolated_work_dir):
        outside = str(Path(isolated_work_dir).parent / "outside.txt")
        with pytest.raises(SandboxViolationError, match="超出工作目录"):
            resolve_sandboxed_path(outside, str(isolated_work_dir))

    def test_rejects_empty_path(self):
        with pytest.raises(ValueError, match="路径不能为空"):
            resolve_sandboxed_path("", "/tmp")

    def test_must_exist(self, isolated_work_dir):
        (isolated_work_dir / "exists.txt").write_text("x", encoding="utf-8")
        result = resolve_sandboxed_path(
            "exists.txt", str(isolated_work_dir), must_exist=True
        )
        assert result.exists()

        with pytest.raises(ValueError, match="路径不存在"):
            resolve_sandboxed_path("missing.txt", str(isolated_work_dir), must_exist=True)

    def test_invalid_work_dir(self):
        with pytest.raises(ValueError, match="工作目录无效"):
            resolve_sandboxed_path("foo.txt", "/nonexistent/path/12345")

    def test_empty_work_dir_uses_cwd(self, isolated_work_dir, monkeypatch):
        monkeypatch.setattr(os, "getcwd", lambda: str(isolated_work_dir))
        result = resolve_sandboxed_path("foo.txt", "")
        assert result == isolated_work_dir / "foo.txt"


class TestResolveSandboxedCwd:
    def test_empty_cwd_returns_work_dir(self, isolated_work_dir):
        result = resolve_sandboxed_cwd("", str(isolated_work_dir))
        assert result == isolated_work_dir.resolve()

    def test_cwd_must_exist(self, isolated_work_dir):
        subdir = isolated_work_dir / "subdir"
        subdir.mkdir()
        result = resolve_sandboxed_cwd("subdir", str(isolated_work_dir))
        assert result == subdir.resolve()

    def test_cwd_outside_work_dir_rejected(self, isolated_work_dir):
        with pytest.raises(SandboxViolationError):
            resolve_sandboxed_cwd("../", str(isolated_work_dir))
