"""持久化配置测试."""

import os
from pathlib import Path

from ai_coding.persistence.config import get_storage_root


class TestGetStorageRoot:
    def test_default_returns_project_data(self):
        root = get_storage_root()
        assert root.name == "data"
        assert "ai-coding" in str(root)

    def test_env_absolute_path(self, monkeypatch, tmp_path):
        env_path = tmp_path / "custom_storage"
        monkeypatch.setenv("AI_CODE_HOME", str(env_path))
        root = get_storage_root()
        assert root == env_path.resolve()

    def test_env_relative_path(self, monkeypatch, tmp_path):
        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            monkeypatch.setenv("AI_CODE_HOME", "relative_storage")
            root = get_storage_root()
            assert root == (tmp_path / "relative_storage").resolve()
        finally:
            os.chdir(original_cwd)

    def test_env_expand_user(self, monkeypatch):
        monkeypatch.setenv("AI_CODE_HOME", "~")
        root = get_storage_root()
        assert root == Path.home().resolve()
