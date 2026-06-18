"""配置管理模块测试."""

import os

import pytest

from ai_coding import config as config_module


class TestGetEnvBool:
    def test_true_values(self, monkeypatch):
        for value in ("true", "True", "1", "yes", "on", "YES"):
            monkeypatch.setenv("TEST_BOOL", value)
            assert config_module.get_env_bool("TEST_BOOL") is True

    def test_false_values(self, monkeypatch):
        for value in ("false", "False", "0", "no", "off", "NO"):
            monkeypatch.setenv("TEST_BOOL", value)
            assert config_module.get_env_bool("TEST_BOOL") is False

    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("TEST_BOOL", raising=False)
        assert config_module.get_env_bool("TEST_BOOL", default=True) is True
        assert config_module.get_env_bool("TEST_BOOL", default=False) is False


class TestGetEnv:
    def test_get_env(self, monkeypatch):
        monkeypatch.setenv("TEST_GET_ENV", "value")
        assert config_module.get_env("TEST_GET_ENV") == "value"
        assert config_module.get_env("TEST_GET_ENV_MISSING", "default") == "default"
