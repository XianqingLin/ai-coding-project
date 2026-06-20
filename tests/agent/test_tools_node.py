"""Tools 节点单元测试.

覆盖工具执行、文件快照更新、Plan 模式约束、edit_proposal 回调、
Todo 与后台任务同步等核心分支.
"""

import os
from typing import Any, Dict, List, Optional

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from ai_coding.agent.nodes._utils import normalize_tool_args
from ai_coding.agent.nodes.tools_node import (
    _extract_plan_path,
    _parse_read_file_result,
    _prompt_plan_approval,
    create_tools_node,
)
from ai_coding.agent.state import AgentState
from ai_coding.tools.base import Tool, ToolParameter, ToolRegistry
from ai_coding.tools.collaboration_tools import AskUserQuestionTool
from ai_coding.tools.plan_tools import EnterPlanModeTool, ExitPlanModeTool
from ai_coding.tools.shell_tools import ExecuteCommandTool
from ai_coding.tools.todo_tool import TodoTool


class _EchoTool(Tool):
    """简单的回显工具，用于常规工具执行测试."""

    name = "echo"
    description = "回显输入"
    requires_approval = False

    @property
    def parameters(self) -> List[ToolParameter]:
        return [ToolParameter("text", "string", "要回显的文本")]

    def execute(self, text: str) -> str:
        return f"echo: {text}"


class _ReadFileTool(Tool):
    """模拟 read_file 返回固定格式结果."""

    name = "read_file"
    description = "读取文件"
    requires_approval = False

    @property
    def parameters(self) -> List[ToolParameter]:
        return [ToolParameter("path", "string", "文件路径")]

    def execute(self, path: str) -> str:
        full_path = os.path.join(self.work_dir, path)
        if not os.path.exists(full_path):
            return f"[错误] 文件不存在: {path}"
        content = open(full_path, "r", encoding="utf-8").read()
        lines = content.split("\n")
        result_lines = [f"文件: {path}", "=" * 20]
        for i, line in enumerate(lines, 1):
            result_lines.append(f"{i:3d} | {line}")
        result_lines.append("=" * 20)
        result_lines.append(f"(本段共 {len(lines)} 行)")
        return "\n".join(result_lines)


class _WriteFileTool(Tool):
    """模拟 write_file."""

    name = "write_file"
    description = "写入文件"
    requires_approval = False

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("path", "string", "文件路径"),
            ToolParameter("content", "string", "文件内容"),
        ]

    def execute(self, path: str, content: str) -> str:
        full_path = os.path.join(self.work_dir, path)
        os.makedirs(os.path.dirname(full_path) or self.work_dir, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[成功] 已写入文件: {path}"


class _EditFileTool(Tool):
    """模拟 edit_file."""

    name = "edit_file"
    description = "编辑文件"
    requires_approval = False

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("path", "string", "文件路径"),
            ToolParameter("old_string", "string", "旧字符串"),
            ToolParameter("new_string", "string", "新字符串"),
        ]

    def execute(self, path: str, old_string: str, new_string: str) -> str:
        full_path = os.path.join(self.work_dir, path)
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        if old_string not in content:
            return f"[错误] 未找到匹配文本: {old_string}"
        content = content.replace(old_string, new_string, 1)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[成功] 已编辑文件: {path}"


def _make_state(
    messages: Optional[List[Any]] = None,
    file_snapshots: Optional[Dict[str, str]] = None,
    todos: Optional[List[dict]] = None,
    background_tasks: Optional[List[dict]] = None,
    plan_mode: bool = False,
    plan_file_path: str = "",
    sub_agents: Optional[List[dict]] = None,
) -> AgentState:
    return {
        "messages": messages or [],
        "file_snapshots": file_snapshots or {},
        "todos": todos or [],
        "globally_approved_tools": [],
        "background_tasks": background_tasks or [],
        "plan_mode": plan_mode,
        "plan_file_path": plan_file_path,
        "sub_agents": sub_agents or [],
    }


@pytest.fixture
def base_registry(isolated_work_dir):
    """创建包含常规测试工具的注册表."""
    registry = ToolRegistry()
    for tool in [
        _EchoTool(),
        _ReadFileTool(),
        _WriteFileTool(),
        _EditFileTool(),
        TodoTool(),
    ]:
        tool.set_work_dir(str(isolated_work_dir))
        registry.register(tool)
    return registry


class TestToolsNodeBasics:
    """Tools 节点基础行为测试."""

    def test_non_ai_message_returns_empty(self, base_registry):
        """最后一条消息不是 AIMessage 时返回空更新."""
        node = create_tools_node(base_registry)
        state = _make_state(messages=[ToolMessage(content="x", tool_call_id="t1")])
        result = node(state)

        assert result["messages"] == []
        assert result["file_snapshots"] == {}

    def test_ai_message_without_tool_calls_returns_empty(self, base_registry):
        """AIMessage 没有 tool_calls 时返回空更新."""
        node = create_tools_node(base_registry)
        state = _make_state(messages=[AIMessage(content="hi")])
        result = node(state)

        assert result["messages"] == []
        assert result["file_snapshots"] == {}

    def test_simple_tool_execution(self, base_registry):
        """常规工具调用生成 ToolMessage."""
        node = create_tools_node(base_registry)
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "echo", "args": {"text": "hello"}}],
        )
        result = node(_make_state(messages=[ai_msg]))

        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "echo: hello"
        assert result["messages"][0].tool_call_id == "tc1"

    def test_rejected_tool_call_is_skipped(self, base_registry):
        """已被拒绝的 tool_call_id 不会再次执行."""
        node = create_tools_node(base_registry)
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "echo", "args": {"text": "hello"}}],
        )
        rejected = ToolMessage(
            content="[系统] 用户拒绝了工具调用 tc1", tool_call_id="tc1"
        )
        state = _make_state(messages=[ai_msg, rejected])
        result = node(state)

        assert result["messages"] == []

    def test_pydantic_args_normalized(self, base_registry):
        """工具节点内部通过 normalize_tool_args 兼容 Pydantic model 参数."""

        class Args(BaseModel):
            text: str

        assert normalize_tool_args(Args(text="pydantic")) == {"text": "pydantic"}
        assert normalize_tool_args({"a": 1}) == {"a": 1}
        assert normalize_tool_args(None) == {}


class TestToolsNodeFileSnapshots:
    """文件快照更新测试."""

    def test_read_file_updates_snapshot(self, base_registry, isolated_work_dir):
        """read_file 执行后更新 file_snapshots."""
        (isolated_work_dir / "a.txt").write_text("line1\nline2", encoding="utf-8")
        node = create_tools_node(base_registry)
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "read_file", "args": {"path": "a.txt"}}],
        )
        result = node(_make_state(messages=[ai_msg]))

        assert result["file_snapshots"].get("a.txt") == "line1\nline2"

    def test_write_file_updates_snapshot_and_emits_proposal(
        self, base_registry, isolated_work_dir
    ):
        """write_file 更新快照并触发 edit_proposal 回调."""
        proposals: List[Dict[str, Any]] = []
        node = create_tools_node(base_registry, on_edit_proposal=proposals.append)
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"path": "b.txt", "content": "new content"},
                }
            ],
        )
        result = node(_make_state(messages=[ai_msg]))

        assert result["file_snapshots"].get("b.txt") == "new content"
        assert len(proposals) == 1
        assert proposals[0]["tool"] == "write_file"
        assert proposals[0]["path"] == "b.txt"
        assert proposals[0]["old_content"] == ""
        assert proposals[0]["new_content"] == "new content"

    def test_edit_file_updates_snapshot(self, base_registry, isolated_work_dir):
        """edit_file 成功后刷新快照."""
        (isolated_work_dir / "c.txt").write_text("old text", encoding="utf-8")
        node = create_tools_node(base_registry)
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "edit_file",
                    "args": {
                        "path": "c.txt",
                        "old_string": "old text",
                        "new_string": "new text",
                    },
                }
            ],
        )
        result = node(_make_state(messages=[ai_msg]))

        assert result["file_snapshots"].get("c.txt") == "new text"

    def test_edit_file_failed_does_not_update_snapshot(
        self, base_registry, isolated_work_dir
    ):
        """edit_file 失败时不应更新快照."""
        (isolated_work_dir / "d.txt").write_text("content", encoding="utf-8")
        node = create_tools_node(base_registry)
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "edit_file",
                    "args": {
                        "path": "d.txt",
                        "old_string": "not exist",
                        "new_string": "x",
                    },
                }
            ],
        )
        result = node(_make_state(messages=[ai_msg]))

        assert "d.txt" not in result["file_snapshots"]


class TestToolsNodeParseHelpers:
    """内部解析 helper 测试."""

    def test_parse_read_file_result(self):
        result = "文件: foo.py\n==========\n  1 | a\n  2 | b\n==========\n(共2行)"
        path, content = _parse_read_file_result(result)
        assert path == "foo.py"
        assert content == "a\nb"

    def test_parse_read_file_result_malformed(self):
        path, content = _parse_read_file_result("not a file header")
        assert path == ""
        assert content == "not a file header"

    def test_extract_plan_path(self):
        result = "[成功] 已进入 Plan 模式。\n计划文件路径: /tmp/plan.md\n其他内容"
        assert _extract_plan_path(result) == "/tmp/plan.md"

    def test_extract_plan_path_missing(self):
        assert _extract_plan_path("no plan here") == ""


class TestToolsNodePlanMode:
    """Plan 模式相关测试."""

    @pytest.fixture
    def plan_registry(self, isolated_work_dir):
        """包含 Plan 工具的注册表."""
        registry = ToolRegistry()
        read_tool = _ReadFileTool()
        read_tool.set_work_dir(str(isolated_work_dir))
        write_tool = _WriteFileTool()
        write_tool.set_work_dir(str(isolated_work_dir))
        edit_tool = _EditFileTool()
        edit_tool.set_work_dir(str(isolated_work_dir))
        enter_tool = EnterPlanModeTool()
        enter_tool.set_work_dir(str(isolated_work_dir))
        exit_tool = ExitPlanModeTool()
        exit_tool.set_work_dir(str(isolated_work_dir))

        registry.register(read_tool)
        registry.register(write_tool)
        registry.register(edit_tool)
        registry.register(enter_tool)
        registry.register(exit_tool)
        return registry

    def test_enter_plan_mode_sets_state(self, plan_registry, isolated_work_dir):
        """enter_plan_mode 工具进入 Plan 模式并返回计划文件路径."""
        node = create_tools_node(plan_registry)
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "enter_plan_mode", "args": {}}],
        )
        result = node(_make_state(messages=[ai_msg]))

        assert result["plan_mode"] is True
        assert result["plan_file_path"].endswith(".md")
        assert (isolated_work_dir / ".kimi" / "plans").exists()

    def test_plan_mode_blocks_write_to_non_plan_file(
        self, plan_registry, isolated_work_dir, monkeypatch
    ):
        """Plan 模式下 write_file 只能写入计划文件."""
        node = create_tools_node(plan_registry)
        plan_path = str(isolated_work_dir / ".kimi" / "plans" / "plan.md")
        # 直接创建计划文件，模拟已进入 Plan 模式
        os.makedirs(os.path.dirname(plan_path), exist_ok=True)
        open(plan_path, "w").close()

        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"path": "other.txt", "content": "x"},
                }
            ],
        )
        state = _make_state(messages=[ai_msg], plan_mode=True, plan_file_path=plan_path)
        result = node(state)

        assert len(result["messages"]) == 1
        assert "只能修改计划文件" in result["messages"][0].content
        assert not (isolated_work_dir / "other.txt").exists()

    def test_plan_mode_allows_write_to_plan_file(
        self, plan_registry, isolated_work_dir
    ):
        """Plan 模式下可以写入计划文件."""
        node = create_tools_node(plan_registry)
        plan_path = str(isolated_work_dir / ".kimi" / "plans" / "plan.md")
        os.makedirs(os.path.dirname(plan_path), exist_ok=True)
        open(plan_path, "w").close()

        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"path": plan_path, "content": "plan content"},
                }
            ],
        )
        state = _make_state(messages=[ai_msg], plan_mode=True, plan_file_path=plan_path)
        result = node(state)

        assert result["messages"][0].content.startswith("[成功]")
        assert open(plan_path, "r").read() == "plan content"

    def test_exit_plan_mode_approve(
        self, plan_registry, isolated_work_dir, monkeypatch
    ):
        """exit_plan_mode 用户批准则退出 Plan 模式."""
        node = create_tools_node(plan_registry)
        plan_path = str(isolated_work_dir / ".kimi" / "plans" / "plan.md")
        os.makedirs(os.path.dirname(plan_path), exist_ok=True)
        with open(plan_path, "w", encoding="utf-8") as f:
            f.write("# My Plan")

        monkeypatch.setattr("builtins.input", lambda _: "a")

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "exit_plan_mode", "args": {}}],
        )
        state = _make_state(messages=[ai_msg], plan_mode=True, plan_file_path=plan_path)
        result = node(state)

        assert result["plan_mode"] is False
        assert result["plan_file_path"] == ""
        assert "已批准" in result["messages"][0].content

    def test_exit_plan_mode_reject_keeps_plan(
        self, plan_registry, isolated_work_dir, monkeypatch
    ):
        """exit_plan_mode 用户拒绝则保持 Plan 模式."""
        node = create_tools_node(plan_registry)
        plan_path = str(isolated_work_dir / ".kimi" / "plans" / "plan.md")
        os.makedirs(os.path.dirname(plan_path), exist_ok=True)
        open(plan_path, "w").close()

        monkeypatch.setattr("builtins.input", lambda _: "r")

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "exit_plan_mode", "args": {}}],
        )
        state = _make_state(messages=[ai_msg], plan_mode=True, plan_file_path=plan_path)
        result = node(state)

        assert result["plan_mode"] is True
        assert result["plan_file_path"] == plan_path
        assert "拒绝了计划" in result["messages"][0].content

    def test_exit_plan_mode_not_in_plan_mode(self, plan_registry):
        """不在 Plan 模式时调用 exit_plan_mode 返回错误."""
        node = create_tools_node(plan_registry)
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "exit_plan_mode", "args": {}}],
        )
        result = node(_make_state(messages=[ai_msg], plan_mode=False))

        assert "当前不在 Plan 模式" in result["messages"][0].content

    def test_exit_plan_mode_with_custom_option(
        self, plan_registry, isolated_work_dir, monkeypatch
    ):
        """exit_plan_mode 选择自定义 option."""
        node = create_tools_node(plan_registry)
        plan_path = str(isolated_work_dir / ".kimi" / "plans" / "plan.md")
        os.makedirs(os.path.dirname(plan_path), exist_ok=True)
        open(plan_path, "w").close()

        monkeypatch.setattr("builtins.input", lambda _: "1")

        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "exit_plan_mode",
                    "args": {
                        "options": [
                            {
                                "label": "方案A",
                                "description": "使用方案 A 实现",
                            }
                        ]
                    },
                }
            ],
        )
        state = _make_state(messages=[ai_msg], plan_mode=True, plan_file_path=plan_path)
        result = node(state)

        assert result["plan_mode"] is False
        assert "方案A" in result["messages"][0].content


class TestToolsNodeAskUser:
    """ask_user_question 工具处理测试."""

    @pytest.fixture
    def ask_registry(self, isolated_work_dir, monkeypatch):
        registry = ToolRegistry()
        ask_tool = AskUserQuestionTool()
        ask_tool.set_work_dir(str(isolated_work_dir))
        registry.register(ask_tool)
        return registry

    def test_ask_user_question(self, ask_registry, monkeypatch):
        """ask_user_question 通过 ToolsNode 执行并返回用户选择."""
        monkeypatch.setattr("builtins.input", lambda _: "1")
        node = create_tools_node(ask_registry)
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "ask_user_question",
                    "args": {
                        "question": "选择语言",
                        "options": [{"label": "Python"}],
                    },
                }
            ],
        )
        result = node(_make_state(messages=[ai_msg]))

        assert len(result["messages"]) == 1
        assert "Python" in result["messages"][0].content


class TestToolsNodeSync:
    """Todo 与后台任务同步测试."""

    def test_todo_sync_from_state(self, isolated_work_dir):
        """ToolsNode 执行前会同步 state 中的 todos 到 TodoTool."""
        registry = ToolRegistry()
        todo_tool = TodoTool()
        todo_tool.set_work_dir(str(isolated_work_dir))
        registry.register(todo_tool)
        node = create_tools_node(registry)

        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "set_todo",
                    "args": {"action": "complete", "index": 1},
                }
            ],
        )
        state = _make_state(
            messages=[ai_msg],
            todos=[{"task": "task1", "done": False}],
        )
        result = node(state)

        assert len(result["todos"]) == 1
        assert result["todos"][0]["done"] is True

    def test_background_task_sync(self, isolated_work_dir):
        """ToolsNode 会同步 state 中的 background_tasks 到 ExecuteCommandTool."""
        registry = ToolRegistry()
        exec_tool = ExecuteCommandTool()
        exec_tool.set_work_dir(str(isolated_work_dir))
        registry.register(exec_tool)
        node = create_tools_node(registry)

        # 先启动一个后台任务，让 exec_tool 内部持有该任务
        exec_tool.execute(
            command="echo hello",
            run_in_background=True,
            description="test background task",
        )
        bg_tasks = exec_tool.background_tasks()
        assert len(bg_tasks) == 1
        tid = bg_tasks[0]["task_id"]

        # 修改 state 中的描述字段，验证同步时不会丢失内部引用
        state_task = dict(bg_tasks[0])
        state_task["description"] = "updated desc"

        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "execute_command",
                    "args": {"command": "echo sync"},
                }
            ],
        )
        state = _make_state(messages=[ai_msg], background_tasks=[state_task])
        result = node(state)

        assert len(result["background_tasks"]) == 1
        assert result["background_tasks"][0]["task_id"] == tid
        assert result["background_tasks"][0]["description"] == "updated desc"


class TestPromptPlanApproval:
    """计划审批交互 helper 测试."""

    def test_approve(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "a")
        assert _prompt_plan_approval("plan", []) == "approve"

    def test_reject(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "r")
        assert _prompt_plan_approval("plan", []) == "reject"

    def test_reject_and_exit(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "x")
        assert _prompt_plan_approval("plan", []) == "reject_and_exit"

    def test_revise(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "v")
        assert _prompt_plan_approval("plan", []) == "revise"

    def test_select_option(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "1")
        assert (
            _prompt_plan_approval("plan", [{"label": "opt1", "description": "d"}])
            == "opt1"
        )

    def test_quit_maps_to_reject_and_exit(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "q")
        assert _prompt_plan_approval("plan", []) == "reject_and_exit"

    def test_eof_maps_to_reject_and_exit(self, monkeypatch):
        def raise_eof(_):
            raise EOFError()

        monkeypatch.setattr("builtins.input", raise_eof)
        assert _prompt_plan_approval("plan", []) == "reject_and_exit"

    def test_invalid_then_valid(self, monkeypatch):
        inputs = iter(["invalid", "a"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        assert _prompt_plan_approval("plan", []) == "approve"
