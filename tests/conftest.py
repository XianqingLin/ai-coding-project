"""Pytest 共享 fixtures."""

import os
import tempfile
from pathlib import Path

import pytest

from ai_coding.agent.sub_agent_manager import SubAgentManager
from ai_coding.tools.shell_tools import ExecuteCommandTool


@pytest.fixture
def isolated_work_dir():
    """创建一个隔离的临时工作目录，用于文件类工具测试.

    Windows 上日志文件句柄可能在测试结束时仍未释放，导致临时目录
    清理失败；使用 ignore_cleanup_errors=True 避免因此阻塞测试.
    """
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(
        prefix="ai_coding_test_", ignore_cleanup_errors=True
    ) as tmp:
        os.chdir(tmp)
        yield Path(tmp)
        os.chdir(original_cwd)


@pytest.fixture
def clean_sub_agent_manager():
    """清理 SubAgentManager 单例，保证测试隔离."""
    SubAgentManager._instance = None
    yield SubAgentManager()
    SubAgentManager._instance = None


@pytest.fixture
def task_manager(isolated_work_dir):
    """创建并配置 ExecuteCommandTool 作为任务管理器."""
    tool = ExecuteCommandTool()
    tool.set_work_dir(str(isolated_work_dir))
    return tool
