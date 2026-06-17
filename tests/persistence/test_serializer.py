"""持久化序列化单元测试."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ai_coding.agent.state import AgentState
from ai_coding.persistence import state_from_json, state_to_json


class TestStateSerializer:
    def test_roundtrip_basic_state(self):
        state: AgentState = {
            "messages": [
                SystemMessage(content="system"),
                HumanMessage(content="hi"),
                AIMessage(content="hello"),
                ToolMessage(content="result", tool_call_id="call_1"),
            ],
            "file_snapshots": {"main.py": "print('hi')"},
            "todos": [{"task": "do something", "done": False}],
            "background_tasks": [],
            "globally_approved_tools": [],
            "plan_mode": False,
            "plan_file_path": "",
            "sub_agents": [],
        }

        json_text = state_to_json(state)
        restored = state_from_json(json_text)

        assert len(restored["messages"]) == 4
        assert isinstance(restored["messages"][0], SystemMessage)
        assert restored["messages"][1].content == "hi"
        assert restored["file_snapshots"]["main.py"] == "print('hi')"
        assert restored["todos"][0]["task"] == "do something"

    def test_roundtrip_with_tool_calls(self):
        state: AgentState = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "id": "call_1",
                        "name": "read_file",
                        "args": {"path": "main.py"},
                    }],
                ),
            ],
            "file_snapshots": {},
            "todos": [],
            "background_tasks": [],
            "globally_approved_tools": [],
            "plan_mode": False,
            "plan_file_path": "",
            "sub_agents": [],
        }

        json_text = state_to_json(state)
        restored = state_from_json(json_text)

        ai_msg = restored["messages"][0]
        assert isinstance(ai_msg, AIMessage)
        assert len(ai_msg.tool_calls) == 1
        assert ai_msg.tool_calls[0]["name"] == "read_file"
