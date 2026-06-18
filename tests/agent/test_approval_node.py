"""审批门控节点测试.

覆盖 approval_node 和 edges 中的权限控制与条件跳转逻辑.
"""

import threading
import time
from concurrent.futures import Future
from typing import Any, Dict, List, Optional

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_coding.agent.nodes.approval_node import create_approval_gate
from ai_coding.agent.nodes.edges import create_should_continue
from ai_coding.agent.state import AgentState
from ai_coding.tools.base import Tool, ToolParameter, ToolRegistry


class _ReadTool(Tool):
    """不需要审批的只读工具."""

    name = "read_file"
    description = "读取文件"
    requires_approval = False

    @property
    def parameters(self) -> List[ToolParameter]:
        return [ToolParameter("path", "string", "文件路径")]

    def execute(self, path: str) -> str:
        return f"read {path}"


class _WriteTool(Tool):
    """需要审批的写操作工具."""

    name = "write_file"
    description = "写入文件"
    requires_approval = True

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("path", "string", "文件路径"),
            ToolParameter("content", "string", "文件内容"),
        ]

    def execute(self, path: str, content: str) -> str:
        return f"wrote {path}"


class _EditTool(Tool):
    """需要审批的编辑工具."""

    name = "edit_file"
    description = "编辑文件"
    requires_approval = True

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("path", "string", "文件路径"),
            ToolParameter("old_string", "string", "旧字符串"),
            ToolParameter("new_string", "string", "新字符串"),
        ]

    def execute(self, path: str, old_string: str, new_string: str) -> str:
        return f"edited {path}"


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """创建包含读/写/编辑工具的注册表."""
    registry = ToolRegistry()
    registry.register(_ReadTool())
    registry.register(_WriteTool())
    registry.register(_EditTool())
    return registry


def _make_state(
    messages: Optional[List[Any]] = None,
    globally_approved_tools: Optional[List[str]] = None,
) -> AgentState:
    return {
        "messages": messages or [],
        "file_snapshots": {},
        "todos": [],
        "globally_approved_tools": globally_approved_tools or [],
        "background_tasks": [],
        "plan_mode": False,
        "plan_file_path": "",
        "sub_agents": [],
    }


class TestApprovalGate:
    """approval_gate 核心行为测试."""

    def test_no_approval_needed_returns_empty(
        self, tool_registry: ToolRegistry
    ) -> None:
        """不需要授权的工具调用直接跳过审批."""
        gate = create_approval_gate(tool_registry)
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"id": "tc1", "name": "read_file", "args": {"path": "foo.txt"}}
            ],
        )
        state = _make_state(messages=[ai_msg])
        result = gate(state)

        assert result["messages"] == []
        assert result["globally_approved_tools"] == []

    def test_non_ai_message_returns_empty(self, tool_registry: ToolRegistry) -> None:
        """最后一条消息不是 AIMessage 时直接返回空."""
        gate = create_approval_gate(tool_registry)
        state = _make_state(messages=[HumanMessage(content="hello")])
        result = gate(state)

        assert result["messages"] == []
        assert result["globally_approved_tools"] == []

    def test_no_tool_calls_returns_empty(self, tool_registry: ToolRegistry) -> None:
        """AIMessage 没有 tool_calls 时直接返回空."""
        gate = create_approval_gate(tool_registry)
        state = _make_state(messages=[AIMessage(content="hi")])
        result = gate(state)

        assert result["messages"] == []
        assert result["globally_approved_tools"] == []

    def test_globally_approved_skips_approval(
        self, tool_registry: ToolRegistry
    ) -> None:
        """已全局授权的工具不再询问."""
        gate = create_approval_gate(tool_registry)
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "write_file", "args": {"path": "a.txt"}}],
        )
        state = _make_state(messages=[ai_msg], globally_approved_tools=["write_file"])
        result = gate(state)

        assert result["messages"] == []
        assert result["globally_approved_tools"] == ["write_file"]

    def test_non_interactive_rejects_write_tool(
        self, tool_registry: ToolRegistry
    ) -> None:
        """非交互模式下直接拒绝需要授权的工具."""
        gate = create_approval_gate(tool_registry, interactive=False)
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "write_file", "args": {"path": "a.txt"}}],
        )
        state = _make_state(messages=[ai_msg])
        result = gate(state)

        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert isinstance(msg, ToolMessage)
        assert "write_file" in str(msg.content)
        assert "非交互模式" in str(msg.content)


class TestApprovalGateCLI:
    """命令行交互审批模式测试."""

    def test_cli_approve_once(
        self, tool_registry: ToolRegistry, monkeypatch: Any
    ) -> None:
        """用户输入 y 同意单次调用."""
        gate = create_approval_gate(tool_registry)
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "write_file", "args": {"path": "a.txt"}}],
        )
        state = _make_state(messages=[ai_msg])

        inputs = iter(["y"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = gate(state)
        assert result["messages"] == []
        assert result["globally_approved_tools"] == []

    def test_cli_reject(self, tool_registry: ToolRegistry, monkeypatch: Any) -> None:
        """用户输入 n 拒绝调用."""
        gate = create_approval_gate(tool_registry)
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "write_file", "args": {"path": "a.txt"}}],
        )
        state = _make_state(messages=[ai_msg])

        inputs = iter(["n"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = gate(state)
        assert len(result["messages"]) == 1
        assert "拒绝" in str(result["messages"][0].content)

    def test_cli_approve_all(
        self, tool_registry: ToolRegistry, monkeypatch: Any
    ) -> None:
        """用户输入 a 全局授权该工具."""
        gate = create_approval_gate(tool_registry)
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "write_file", "args": {"path": "a.txt"}}],
        )
        state = _make_state(messages=[ai_msg])

        inputs = iter(["a"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = gate(state)
        assert result["messages"] == []
        assert "write_file" in result["globally_approved_tools"]

    def test_cli_quit(self, tool_registry: ToolRegistry, monkeypatch: Any) -> None:
        """用户输入 q 退出并拒绝."""
        gate = create_approval_gate(tool_registry)
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "write_file", "args": {"path": "a.txt"}}],
        )
        state = _make_state(messages=[ai_msg])

        inputs = iter(["q"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = gate(state)
        assert len(result["messages"]) == 1
        assert "退出" in str(result["messages"][0].content)

    def test_cli_eof_as_reject(
        self, tool_registry: ToolRegistry, monkeypatch: Any
    ) -> None:
        """EOFError 视为拒绝."""
        gate = create_approval_gate(tool_registry)
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "write_file", "args": {"path": "a.txt"}}],
        )
        state = _make_state(messages=[ai_msg])

        def raise_eof(_: str) -> str:
            raise EOFError()

        monkeypatch.setattr("builtins.input", raise_eof)

        result = gate(state)
        assert len(result["messages"]) == 1


class TestApprovalGateFrontend:
    """前端驱动审批模式测试."""

    def _run_frontend_approval(
        self,
        tool_registry: ToolRegistry,
        choice: str,
    ) -> Dict[str, Any]:
        """在前端驱动模式下运行审批门控并返回结果."""
        requests: List[Dict[str, Any]] = []
        pending_futures: Dict[str, Future[str]] = {}

        def on_request(request: Dict[str, Any]) -> None:
            requests.append(request)

        def register_future(request_id: str, future: Future[str]) -> Future[str]:
            pending_futures[request_id] = future
            return future

        gate = create_approval_gate(
            tool_registry,
            event_loop="dummy_loop",
            on_approval_request=on_request,
            register_approval_future=register_future,
        )
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "write_file", "args": {"path": "a.txt"}}],
        )
        state = _make_state(messages=[ai_msg])

        result_holder: Dict[str, Any] = {}

        def gate_thread() -> None:
            result_holder["result"] = gate(state)

        t = threading.Thread(target=gate_thread)
        t.start()

        # 等待审批请求发出并获取 future
        for _ in range(100):
            if pending_futures:
                break
            time.sleep(0.01)

        assert pending_futures, "未收到审批请求"
        future = next(iter(pending_futures.values()))
        future.set_result(choice)
        t.join(timeout=2)

        return result_holder["result"]

    def test_frontend_approve(self, tool_registry: ToolRegistry) -> None:
        """前端返回 approve 同意单次调用."""
        result = self._run_frontend_approval(tool_registry, "approve")
        assert len(result["messages"]) == 0

    def test_frontend_approve_all(self, tool_registry: ToolRegistry) -> None:
        """前端返回 approve_all 全局授权."""
        result = self._run_frontend_approval(tool_registry, "approve_all")
        assert "write_file" in result["globally_approved_tools"]

    def test_frontend_reject(self, tool_registry: ToolRegistry) -> None:
        """前端返回 reject 拒绝调用."""
        result = self._run_frontend_approval(tool_registry, "reject")
        assert len(result["messages"]) == 1
        assert "拒绝" in str(result["messages"][0].content)


class TestShouldContinue:
    """条件边 should_continue 测试."""

    def test_end_when_no_tool_calls(self, tool_registry: ToolRegistry) -> None:
        """没有 tool_calls 时返回 END."""
        should_continue = create_should_continue(tool_registry)
        state = _make_state(messages=[AIMessage(content="hello")])
        assert should_continue(state) == "__end__"

    def test_tools_when_no_approval_needed(self, tool_registry: ToolRegistry) -> None:
        """只有不需要授权的工具时返回 tools."""
        should_continue = create_should_continue(tool_registry)
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"id": "tc1", "name": "read_file", "args": {"path": "foo.txt"}}
            ],
        )
        state = _make_state(messages=[ai_msg])
        assert should_continue(state) == "tools"

    def test_approval_when_write_tool_present(
        self, tool_registry: ToolRegistry
    ) -> None:
        """有需要授权的工具时返回 approval."""
        should_continue = create_should_continue(tool_registry)
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"id": "tc1", "name": "read_file", "args": {"path": "foo.txt"}},
                {"id": "tc2", "name": "write_file", "args": {"path": "a.txt"}},
            ],
        )
        state = _make_state(messages=[ai_msg])
        assert should_continue(state) == "approval"

    def test_tools_when_globally_approved(self, tool_registry: ToolRegistry) -> None:
        """工具已全局授权时返回 tools."""
        should_continue = create_should_continue(tool_registry)
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "write_file", "args": {"path": "a.txt"}}],
        )
        state = _make_state(messages=[ai_msg], globally_approved_tools=["write_file"])
        assert should_continue(state) == "tools"

    def test_tools_when_auto_approve(self, tool_registry: ToolRegistry) -> None:
        """auto_approve=True 时直接返回 tools."""
        should_continue = create_should_continue(tool_registry, auto_approve=True)
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "write_file", "args": {"path": "a.txt"}}],
        )
        state = _make_state(messages=[ai_msg])
        assert should_continue(state) == "tools"
