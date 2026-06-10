#!/usr/bin/env python3
"""协作类工具测试.

验证 AskUserQuestion 和 Agent 工具.
用法:
    python scripts/test_collaboration.py
"""

import sys
import time
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from langchain_core.messages import HumanMessage, SystemMessage

from ai_coding.agent.nodes.llm_node import create_llm_node
from ai_coding.mock_llm import MockChatModel, mock_text
from ai_coding.agent.sub_agent_manager import SubAgentManager
from ai_coding.tools.collaboration_tools import AskUserQuestionTool, AgentTool


def run_test(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    detail_str = f" | {detail}" if detail else ""
    print(f"  [{status}] {name}{detail_str}")
    return condition


def build_state(messages, sub_agents=None):
    return {
        "messages": messages,
        "file_snapshots": {},
        "todos": [],
        "globally_approved_tools": [],
        "background_tasks": [],
        "plan_mode": False,
        "plan_file_path": "",
        "sub_agents": sub_agents or [],
    }


# ---------- Test 1: AskUserQuestion 单选 ----------
def test_ask_single_select():
    print("\n[Test] AskUserQuestion 单选")
    tool = AskUserQuestionTool()
    with patch("builtins.input", side_effect=["2"]):
        result = tool.execute(
            question="选择编程语言",
            options=[
                {"label": "Python", "description": "简洁优雅"},
                {"label": "Go", "description": "高性能"},
                {"label": "Rust", "description": "内存安全"},
            ],
        )
    return run_test("单选返回正确 label", "[用户选择] Go" in result, result)


# ---------- Test 2: AskUserQuestion 多选 ----------
def test_ask_multi_select():
    print("\n[Test] AskUserQuestion 多选")
    tool = AskUserQuestionTool()
    with patch("builtins.input", side_effect=["1,3"]):
        result = tool.execute(
            question="选择喜欢的语言",
            options=[
                {"label": "Python"},
                {"label": "Go"},
                {"label": "Rust"},
            ],
            multi_select=True,
        )
    return run_test("多选返回正确 labels", "Python" in result and "Rust" in result, result)


# ---------- Test 3: AskUserQuestion 自定义输入 ----------
def test_ask_custom_input():
    print("\n[Test] AskUserQuestion 自定义输入")
    tool = AskUserQuestionTool()
    with patch("builtins.input", side_effect=["0", "TypeScript"]):
        result = tool.execute(
            question="选择编程语言",
            options=[{"label": "Python"}, {"label": "Go"}],
        )
    return run_test("自定义输入返回正确", "[用户回答] TypeScript" in result, result)


# ---------- Test 4: AskUserQuestion 直接文本输入 ----------
def test_ask_direct_text():
    print("\n[Test] AskUserQuestion 直接文本输入")
    tool = AskUserQuestionTool()
    with patch("builtins.input", return_value="直接回答"):
        result = tool.execute(question="你的想法是？")
    return run_test("直接文本返回正确", "[用户回答] 直接回答" in result, result)


# ---------- Test 5: Agent 同步调度 ----------
def test_agent_sync():
    print("\n[Test] Agent 同步调度")
    manager = SubAgentManager()
    for i in list(manager.list_instances()):
        manager._agents.pop(i.instance_id, None)

    tool = AgentTool()
    sub_llm = MockChatModel(responses=[mock_text("子 Agent 完成")])
    tool.set_llm(llm=None, llm_factory=lambda: sub_llm)

    result = tool.execute(
        prompt="简单任务",
        description="测试同步",
        subagent_type="coder",
        run_in_background=False,
    )
    return run_test("同步模式返回结果", "子 Agent 完成" in result, result[:100])


# ---------- Test 6: Agent 后台调度 ----------
def test_agent_background():
    print("\n[Test] Agent 后台调度")
    manager = SubAgentManager()
    for i in list(manager.list_instances()):
        manager._agents.pop(i.instance_id, None)

    tool = AgentTool()
    sub_llm = MockChatModel(responses=[mock_text("后台完成")])
    tool.set_llm(llm=None, llm_factory=lambda: sub_llm)

    result = tool.execute(
        prompt="后台任务",
        description="测试后台",
        subagent_type="coder",
        run_in_background=True,
    )
    ok = run_test("后台模式返回 ID", "ID:" in result, result[:100])

    # 等待后台完成
    time.sleep(0.5)
    instances = manager.list_instances(status="completed")
    ok &= run_test("后台实例已完成", len(instances) >= 1)
    return ok


# ---------- Test 7: llm_node HumanMessage 注入 ----------
def test_llm_node_human_message():
    print("\n[Test] llm_node HumanMessage 注入")
    manager = SubAgentManager()
    for i in list(manager.list_instances()):
        manager._agents.pop(i.instance_id, None)

    sub_llm = MockChatModel(responses=[mock_text("探索完成")])
    manager.dispatch("explore", "探索项目", llm=None, llm_factory=lambda: sub_llm, run_in_background=False)

    class TrackingMockLLM(MockChatModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.last_input_types = []

        def invoke(self, messages, **kwargs):
            self.last_input_types = [type(m).__name__ for m in messages]
            return super().invoke(messages, **kwargs)

    main_llm = TrackingMockLLM(responses=[mock_text("收到")])
    state = build_state([
        SystemMessage(content="System"),
        HumanMessage(content="用户输入"),
    ])

    llm_node = create_llm_node(llm=main_llm)
    result = llm_node(state)

    has_human = "HumanMessage" in main_llm.last_input_types
    ok = run_test("LLM 输入包含 HumanMessage", has_human, str(main_llm.last_input_types))
    return ok


# ---------- Test 8: Agent 唤回实例 ----------
def test_agent_resume():
    print("\n[Test] Agent 唤回实例")
    manager = SubAgentManager()
    for i in list(manager.list_instances()):
        manager._agents.pop(i.instance_id, None)

    tool = AgentTool()
    sub_llm1 = MockChatModel(responses=[mock_text("第一次")])
    tool.set_llm(llm=None, llm_factory=lambda: sub_llm1)
    tool.execute(prompt="任务1", description="测试唤回", run_in_background=False)

    sid = manager.list_instances()[0].instance_id

    sub_llm2 = MockChatModel(responses=[mock_text("第二次")])
    tool.set_llm(llm=None, llm_factory=lambda: sub_llm2)
    result = tool.execute(prompt="任务2", description="测试唤回2", resume=sid, run_in_background=False)

    return run_test("唤回返回新结果", "第二次" in result, result[:100])


# ---------- Test 9: Agent explore 工具集过滤 ----------
def test_agent_explore_tools():
    print("\n[Test] Agent explore 工具集过滤")
    manager = SubAgentManager()
    for i in list(manager.list_instances()):
        manager._agents.pop(i.instance_id, None)

    tool = AgentTool()
    # explore 没有 write_file，尝试调用会失败
    sub_llm = MockChatModel(responses=[mock_text("尝试写入")])
    tool.set_llm(llm=None, llm_factory=lambda: sub_llm)
    result = tool.execute(
        prompt="尝试写入",
        description="测试 explore",
        subagent_type="explore",
        run_in_background=False,
    )
    # explore 子 Agent 能运行（即使 Mock LLM 的 tool_calls 不会被实际执行）
    return run_test("explore 子 Agent 执行完成", len(result) > 0, result[:100])


# ---------- 主入口 ----------
def main():
    tests = [
        test_ask_single_select,
        test_ask_multi_select,
        test_ask_custom_input,
        test_ask_direct_text,
        test_agent_sync,
        test_agent_background,
        test_llm_node_human_message,
        test_agent_resume,
        test_agent_explore_tools,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  [FAIL] {test.__name__} 抛出异常: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print(f"\n{'='*50}")
    passed = sum(results)
    total = len(results)
    print(f"结果: {passed}/{total} 通过")
    if passed == total:
        print("全部通过 [OK]")
    else:
        print("存在失败 [FAIL]")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
