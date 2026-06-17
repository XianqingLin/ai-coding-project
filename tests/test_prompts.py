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

    tool_guidelines = files("ai_coding.prompts").joinpath("guidelines_tool_usage.txt").read_text(encoding="utf-8")
    system_guidelines = files("ai_coding.prompts").joinpath("guidelines_system_behavior.txt").read_text(encoding="utf-8")

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
