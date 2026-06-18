"""协作类工具测试."""

from ai_coding.agent.sub_agent_manager import SubAgentManager
from ai_coding.mock_llm import MockChatModel, mock_text
from ai_coding.tools.collaboration_tools import AgentTool, AskUserQuestionTool


class TestAskUserQuestionTool:
    """AskUserQuestionTool 测试."""

    def test_empty_question_returns_error(self):
        """空 question 应返回错误."""
        tool = AskUserQuestionTool()
        result = tool.execute(question="")
        assert "错误" in result

    def test_single_select_valid_option(self, monkeypatch):
        """单选有效选项."""
        monkeypatch.setattr("builtins.input", lambda _: "2")
        tool = AskUserQuestionTool()
        result = tool.execute(
            question="选择颜色",
            options=[{"label": "红"}, {"label": "蓝"}],
        )
        assert "[用户选择] 蓝" in result

    def test_single_select_custom_input(self, monkeypatch):
        """选择 0 后输入自定义回答."""
        inputs = iter(["0", "自定义答案"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        tool = AskUserQuestionTool()
        result = tool.execute(
            question="选择或输入",
            options=[{"label": "A"}],
        )
        assert "[用户回答] 自定义答案" in result

    def test_multi_select(self, monkeypatch):
        """多选有效选项."""
        monkeypatch.setattr("builtins.input", lambda _: "1,3")
        tool = AskUserQuestionTool()
        result = tool.execute(
            question="多选",
            options=[{"label": "A"}, {"label": "B"}, {"label": "C"}],
            multi_select=True,
        )
        assert "[用户选择] A, C" in result

    def test_direct_text_input(self, monkeypatch):
        """非数字输入视为直接文本回答."""
        monkeypatch.setattr("builtins.input", lambda _: "直接回答")
        tool = AskUserQuestionTool()
        result = tool.execute(question="你的意见？")
        assert "[用户回答] 直接回答" in result


class TestAgentTool:
    """AgentTool 测试."""

    def test_empty_prompt_returns_error(self):
        """空 prompt 应返回错误."""
        tool = AgentTool()
        result = tool.execute(prompt="", description="test")
        assert "错误" in result

    def test_empty_description_returns_error(self):
        """空 description 应返回错误."""
        tool = AgentTool()
        result = tool.execute(prompt="do something", description="")
        assert "错误" in result

    def test_unsupported_subagent_type(self):
        """不支持的 subagent_type 应返回错误."""
        tool = AgentTool()
        tool.set_llm(MockChatModel())
        result = tool.execute(
            prompt="do something",
            description="test",
            subagent_type="unknown",
        )
        assert "错误" in result

    def test_missing_llm_returns_error(self):
        """未设置 LLM 应返回错误."""
        tool = AgentTool()
        result = tool.execute(prompt="do something", description="test")
        assert "错误" in result
        assert "LLM" in result

    def test_sync_dispatch(
        self, clean_sub_agent_manager, isolated_work_dir, monkeypatch
    ):
        """前台委派子 Agent 应返回执行结果."""
        SubAgentManager._instance = None
        tool = AgentTool()
        tool.set_parent_context(work_dir=str(isolated_work_dir))
        tool.set_llm(MockChatModel(responses=[mock_text("sub result")]))

        result = tool.execute(
            prompt="read file",
            description="test sub agent",
            subagent_type="coder",
        )

        assert "sub result" in result
        SubAgentManager._instance = None

    def test_background_dispatch(self, clean_sub_agent_manager, isolated_work_dir):
        """后台委派子 Agent 应返回任务 ID."""
        SubAgentManager._instance = None
        tool = AgentTool()
        tool.set_parent_context(work_dir=str(isolated_work_dir))
        tool.set_llm(MockChatModel(responses=[mock_text("done")]))

        result = tool.execute(
            prompt="background task",
            description="test background",
            subagent_type="coder",
            run_in_background=True,
        )

        assert "后台启动" in result
        SubAgentManager._instance = None
