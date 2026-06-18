"""系统提示词构建测试."""

from importlib.resources import files

from ai_coding.agent.core import LangGraphAgent
from ai_coding.environment import EnvironmentInfo
from ai_coding.mock_llm import MockChatModel
from ai_coding.prompts import PromptContext, SystemPromptBuilder


def _make_prompt_context() -> PromptContext:
    env_info = EnvironmentInfo(
        cwd="/workspace",
        is_git_repo=True,
        platform="TestOS",
        shell="/bin/test",
        os_version="TestOS-1.0",
        marketing_name="Test Model",
        model_id="test-model",
        knowledge_cutoff="2099-01",
        date="2099-01-01",
    )
    return PromptContext(environment_info=env_info)


def test_system_prompt_builder_basic() -> None:
    """SystemPromptBuilder 应包含所有静态和动态内容."""
    context = _make_prompt_context()
    builder = SystemPromptBuilder()
    prompt = builder.build(context)

    assert "你是一个 AI 编程助手" in prompt
    assert "## 工具使用规范" in prompt
    assert "## 任务执行规范" in prompt
    assert "## 系统交互规范" in prompt
    assert "# 环境" in prompt
    assert "主工作目录：/workspace" in prompt
    assert "日期：2099-01-01" in prompt


def test_system_prompt_builder_without_context() -> None:
    """SystemPromptBuilder 在无上下文时仍能构建静态 system prompt."""
    builder = SystemPromptBuilder()
    prompt = builder.build()

    assert "你是一个 AI 编程助手" in prompt
    assert "## 工具使用规范" in prompt
    assert "# 环境" not in prompt
    assert "日期：" not in prompt


def test_langgraph_agent_uses_prompt_context() -> None:
    """LangGraphAgent 应通过 PromptContext 构建 system prompt."""
    context = _make_prompt_context()
    agent = LangGraphAgent(
        llm_factory=lambda: MockChatModel(),
        prompt_context=context,
    )

    assert "主工作目录：/workspace" in agent.system_prompt
    assert "日期：2099-01-01" in agent.system_prompt
    assert "## 系统交互规范" in agent.system_prompt


def test_system_prompt_structure() -> None:
    """system prompt 的结构应为：主模板 -> 行为准则 -> 环境信息 -> 日期."""
    context = _make_prompt_context()
    agent = LangGraphAgent(
        llm_factory=lambda: MockChatModel(),
        prompt_context=context,
    )

    tool_guidelines = (
        files("ai_coding.prompts")
        .joinpath("guidelines_tool_usage.txt")
        .read_text(encoding="utf-8")
    )
    system_guidelines = (
        files("ai_coding.prompts")
        .joinpath("guidelines_system_behavior.txt")
        .read_text(encoding="utf-8")
    )

    main_pos = agent.system_prompt.find("你是一个 AI 编程助手")
    tool_pos = agent.system_prompt.find(tool_guidelines.splitlines()[0])
    system_pos = agent.system_prompt.find(system_guidelines.splitlines()[0])
    env_pos = agent.system_prompt.find("# 环境")
    date_pos = agent.system_prompt.find("日期：")

    assert main_pos != -1
    assert tool_pos != -1
    assert system_pos != -1
    assert env_pos != -1
    assert date_pos != -1
    assert main_pos < tool_pos < system_pos < env_pos < date_pos


def test_system_prompt_builder_missing_guideline_file() -> None:
    """当传入不存在的 guideline 文件时，应抛出 FileNotFoundError，并包含文件名."""
    import pytest

    builder = SystemPromptBuilder(guideline_files=["nonexistent_guideline.txt"])

    with pytest.raises(FileNotFoundError) as exc_info:
        builder.build()

    assert "nonexistent_guideline.txt" in str(exc_info.value)


def test_prompt_context_renders_memory_and_language() -> None:
    """PromptContext 应渲染记忆摘要和语言偏好."""
    context = PromptContext(
        memory_summary="这是之前的记忆摘要",
        language="中文",
    )
    parts = context.render_dynamic_parts()

    assert any("记忆摘要" in p for p in parts)
    assert any("中文" in p for p in parts)


def test_prompt_context_empty_renders_nothing() -> None:
    """PromptContext 所有字段为空时不渲染任何动态段落."""
    context = PromptContext()
    assert context.render_dynamic_parts() == []


def test_prompt_context_no_environment_info() -> None:
    """environment_info 为 None 时环境和日期都不渲染."""
    context = PromptContext(environment_info=None)
    assert context._render_environment() == ""
    assert context._render_date() == ""
