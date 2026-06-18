"""环境信息收集与注入测试."""

import re

from ai_coding.agent.service import AgentService
from ai_coding.environment import EnvironmentInfo, collect_environment_info


def test_environment_info_render() -> None:
    """EnvironmentInfo.render() 输出应包含所有字段."""
    info = EnvironmentInfo(
        cwd="/workspace",
        is_git_repo=True,
        platform="Windows",
        shell="cmd.exe",
        os_version="Windows-10",
        marketing_name="Kimi",
        model_id="kimi-k2.6",
        knowledge_cutoff="2025-01",
        date="2026-06-16",
    )
    text = info.render()

    assert "主工作目录：/workspace" in text
    assert "是否为 git 仓库：是" in text
    assert "平台：Windows" in text
    assert "Shell：cmd.exe" in text
    assert "操作系统版本：Windows-10" in text
    assert "你由名为 Kimi 的模型驱动" in text
    assert "确切的模型 ID 是 kimi-k2.6" in text
    assert "助手知识截止日期为 2025-01" in text


def test_environment_info_to_dict() -> None:
    """EnvironmentInfo.to_dict() 应返回用于格式化主模板的字段."""
    info = EnvironmentInfo(
        cwd="/workspace",
        is_git_repo=True,
        platform="Windows",
        shell="cmd.exe",
        os_version="Windows-10",
        marketing_name="Kimi",
        model_id="kimi-k2.6",
        knowledge_cutoff="2025-01",
        date="2026-06-16",
    )
    d = info.to_dict()

    assert d == {"cwd": "/workspace", "date": "2026-06-16"}


def test_collect_environment_info_basic(isolated_work_dir) -> None:
    """collect_environment_info 应返回正确类型和字段."""
    info = collect_environment_info(str(isolated_work_dir), provider="mock")

    assert isinstance(info, EnvironmentInfo)
    assert info.cwd == str(isolated_work_dir.resolve())
    assert info.platform != ""
    assert info.shell != ""
    assert info.os_version != ""
    assert info.marketing_name == "Mock Model"
    assert info.model_id == "mock"
    assert info.knowledge_cutoff == "N/A"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", info.date)


def test_collect_environment_info_detects_git_repo(isolated_work_dir) -> None:
    """collect_environment_info 应能检测 git 仓库."""
    info = collect_environment_info(str(isolated_work_dir), provider="mock")
    assert info.is_git_repo is False

    git_dir = isolated_work_dir / ".git"
    git_dir.mkdir()

    info = collect_environment_info(str(isolated_work_dir), provider="mock")
    assert info.is_git_repo is True


def test_agentservice_injects_environment_info(isolated_work_dir) -> None:
    """AgentService 默认应在 system prompt 中注入环境信息."""
    service = AgentService(
        work_dir=str(isolated_work_dir),
        llm_provider="mock",
        enable_env_info=True,
    )
    agent = service._get_agent()

    assert agent is not None
    system_prompt = agent.system_prompt

    # cwd 出现在环境信息段落中
    assert "主工作目录" in system_prompt
    assert str(isolated_work_dir.resolve()) in system_prompt
    assert "是否为 git 仓库" in system_prompt
    assert "# 环境" in system_prompt


def test_agentservice_can_disable_environment_info(isolated_work_dir) -> None:
    """AgentService 可关闭环境信息注入."""
    service = AgentService(
        work_dir=str(isolated_work_dir),
        llm_provider="mock",
        enable_env_info=False,
    )
    agent = service._get_agent()

    assert agent is not None
    system_prompt = agent.system_prompt

    assert "主工作目录" not in system_prompt
    assert "# 环境" not in system_prompt
