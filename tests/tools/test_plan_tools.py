"""Plan 模式工具测试."""

from pathlib import Path

from ai_coding.tools.plan_tools import EnterPlanModeTool, ExitPlanModeTool


class TestEnterPlanModeTool:
    """进入 Plan 模式工具测试."""

    def test_enter_plan_mode_creates_plan_file(self, isolated_work_dir):
        """进入 Plan 模式应创建计划文件."""
        tool = EnterPlanModeTool()
        tool.set_work_dir(str(isolated_work_dir))

        result = tool.execute()

        assert "成功" in result.data
        assert "已进入 Plan 模式" in result.data
        # 计划文件应存在
        plan_dir = Path(isolated_work_dir) / ".kimi" / "plans"
        assert plan_dir.exists()
        files = list(plan_dir.glob("plan_*.md"))
        assert len(files) == 1
        assert files[0].read_text(encoding="utf-8").startswith("# Plan")

    def test_enter_plan_mode_returns_path(self, isolated_work_dir):
        """返回结果应包含计划文件路径."""
        tool = EnterPlanModeTool()
        tool.set_work_dir(str(isolated_work_dir))

        result = tool.execute()

        assert "计划文件路径:" in result.data


class TestExitPlanModeTool:
    """退出 Plan 模式工具测试."""

    def test_exit_without_options(self, isolated_work_dir):
        """无 options 时应通过校验."""
        tool = ExitPlanModeTool()
        tool.set_work_dir(str(isolated_work_dir))

        result = tool.execute()

        # 当前实现返回空字符串，这是已知行为
        assert result.data == ""

    def test_exit_with_valid_options(self, isolated_work_dir):
        """有效 options 应通过校验."""
        tool = ExitPlanModeTool()
        tool.set_work_dir(str(isolated_work_dir))

        result = tool.execute(options=[{"label": "方案 A", "description": "快速实现"}])

        # 当前实现返回空字符串
        assert result.data == ""

    def test_exit_with_too_many_options(self, isolated_work_dir):
        """options 超过 3 个应返回错误."""
        tool = ExitPlanModeTool()
        tool.set_work_dir(str(isolated_work_dir))

        options = [
            {"label": "A", "description": ""},
            {"label": "B", "description": ""},
            {"label": "C", "description": ""},
            {"label": "D", "description": ""},
        ]
        result = tool.execute(options=options)

        assert "错误" in result.data
        assert "最多" in result.data

    def test_exit_with_reserved_label(self, isolated_work_dir):
        """使用保留词 label 应返回错误."""
        tool = ExitPlanModeTool()
        tool.set_work_dir(str(isolated_work_dir))

        result = tool.execute(options=[{"label": "approve", "description": ""}])

        assert "错误" in result.data
        assert "保留词" in result.data

    def test_exit_with_duplicate_labels(self, isolated_work_dir):
        """重复 label 应返回错误."""
        tool = ExitPlanModeTool()
        tool.set_work_dir(str(isolated_work_dir))

        result = tool.execute(
            options=[
                {"label": "方案 A", "description": ""},
                {"label": "方案 A", "description": ""},
            ]
        )

        assert "错误" in result.data
        assert "重复" in result.data

    def test_exit_with_long_label(self, isolated_work_dir):
        """label 超过 80 字符应返回错误."""
        tool = ExitPlanModeTool()
        tool.set_work_dir(str(isolated_work_dir))

        result = tool.execute(options=[{"label": "x" * 81, "description": ""}])

        assert "错误" in result.data
        assert "80" in result.data

    def test_exit_with_invalid_option_type(self, isolated_work_dir):
        """options 元素非字典应返回错误."""
        tool = ExitPlanModeTool()
        tool.set_work_dir(str(isolated_work_dir))

        result = tool.execute(options=["not a dict"])

        assert "错误" in result.data
