#!/usr/bin/env python3
"""Plan 模式功能测试.

验证 EnterPlanMode / ExitPlanMode 工具及 Plan 模式约束。
用法:
    python scripts/test_plan_mode.py
"""

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from langchain_core.messages import AIMessage

from ai_coding.agent.nodes.tools_node import create_tools_node
from ai_coding.mock_llm import mock_text
from ai_coding.tools import DEFAULT_TOOLS, ToolRegistry


def build_state(tool_calls, plan_mode=False, plan_file_path="", extra_msgs=None):
    """构建测试用的 AgentState."""
    msgs = [mock_text("start")]
    if extra_msgs:
        msgs.extend(extra_msgs)
    msgs.append(AIMessage(
        content="",
        tool_calls=[{
            "name": tc.name,
            "args": tc.args,
            "id": tc.tool_call_id,
            "type": "tool_call",
        } for tc in tool_calls],
    ))
    return {
        "messages": msgs,
        "file_snapshots": {},
        "todos": [],
        "globally_approved_tools": [],
        "background_tasks": [],
        "plan_mode": plan_mode,
        "plan_file_path": plan_file_path,
    }


def setup_registry():
    reg = ToolRegistry()
    for tool in DEFAULT_TOOLS:
        reg.register(tool)
    return reg


def run_test(name, state, expected_plan_mode, expected_plan_file_path=None,
             check_callback=None):
    """运行单条测试并断言 plan_mode / plan_file_path.

    expected_plan_file_path 为 None 表示不检查该字段。
    """
    print(f"\n{'='*50}")
    print(f"[TEST] {name}")
    reg = setup_registry()
    node = create_tools_node(reg)
    result = node(state)

    actual_pm = result.get("plan_mode", False)
    actual_pfp = result.get("plan_file_path", "")

    ok = True
    if actual_pm != expected_plan_mode:
        print(f"  FAIL: plan_mode 期望={expected_plan_mode}, 实际={actual_pm}")
        ok = False
    if expected_plan_file_path is not None and actual_pfp != expected_plan_file_path:
        print(f"  FAIL: plan_file_path 期望={expected_plan_file_path!r}, 实际={actual_pfp!r}")
        ok = False

    msgs = result.get("messages", [])
    if check_callback:
        cb_ok = check_callback(msgs)
        if not cb_ok:
            ok = False

    if ok:
        print(f"  PASS")
    return ok


class MockTC:
    """轻量 tool_call 包装."""
    def __init__(self, name, args, call_id=""):
        self.name = name
        self.args = args
        self.tool_call_id = call_id or f"tc_{name}"


def main():
    results = []

    # ---------- Test 1: enter_plan_mode ----------
    def check_enter(msgs):
        if not msgs:
            print("  FAIL: 没有返回 ToolMessage")
            return False
        content = msgs[0].content
        if "计划文件路径:" not in content:
            print(f"  FAIL: 返回内容缺少路径: {content[:200]!r}")
            return False
        return True

    state1 = build_state([MockTC("enter_plan_mode", {})])
    results.append(run_test(
        "enter_plan_mode 设置 plan_mode=True",
        state1,
        expected_plan_mode=True,
        check_callback=check_enter,
    ))

    # ---------- Test 2: Plan 模式下 write_file 到非法路径被拦截 ----------
    # 先获取一个真实的计划文件路径
    reg = setup_registry()
    enter_result = reg.execute("enter_plan_mode", {})
    plan_path = None
    for line in enter_result.split("\n"):
        if line.startswith("计划文件路径:"):
            plan_path = line.replace("计划文件路径:", "").strip()
            break

    assert plan_path, "无法获取计划文件路径"

    def check_write_blocked(msgs):
        if not msgs:
            print("  FAIL: 没有返回消息")
            return False
        content = msgs[0].content
        if "Plan 模式下只能修改计划文件" not in content:
            print(f"  FAIL: 未拦截 write_file: {content[:200]!r}")
            return False
        return True

    state2 = build_state(
        [MockTC("write_file", {"path": "other.py", "content": "x"})],
        plan_mode=True,
        plan_file_path=plan_path,
    )
    results.append(run_test(
        "Plan 模式下 write_file 到非法路径被拦截",
        state2,
        expected_plan_mode=True,
        expected_plan_file_path=plan_path,
        check_callback=check_write_blocked,
    ))

    # ---------- Test 3: Plan 模式下 edit_file 到非法路径被拦截 ----------
    def check_edit_blocked(msgs):
        content = msgs[0].content
        if "Plan 模式下只能修改计划文件" not in content:
            print(f"  FAIL: 未拦截 edit_file: {content[:200]!r}")
            return False
        return True

    state3 = build_state(
        [MockTC("edit_file", {"path": "other.py", "old_string": "a", "new_string": "b"})],
        plan_mode=True,
        plan_file_path=plan_path,
    )
    results.append(run_test(
        "Plan 模式下 edit_file 到非法路径被拦截",
        state3,
        expected_plan_mode=True,
        expected_plan_file_path=plan_path,
        check_callback=check_edit_blocked,
    ))

    # ---------- Test 4: Plan 模式下 task_stop 被拦截 ----------
    def check_task_stop_blocked(msgs):
        content = msgs[0].content
        if "Plan 模式下不能使用 task_stop" not in content:
            print(f"  FAIL: 未拦截 task_stop: {content[:200]!r}")
            return False
        return True

    state4 = build_state(
        [MockTC("task_stop", {"task_id": "t1"})],
        plan_mode=True,
        plan_file_path=plan_path,
    )
    results.append(run_test(
        "Plan 模式下 task_stop 被拦截",
        state4,
        expected_plan_mode=True,
        expected_plan_file_path=plan_path,
        check_callback=check_task_stop_blocked,
    ))

    # ---------- Test 5: Plan 模式下 write_file 到计划文件允许 ----------
    state5 = build_state(
        [MockTC("write_file", {"path": plan_path, "content": "# New Plan\n\n"})],
        plan_mode=True,
        plan_file_path=plan_path,
    )
    results.append(run_test(
        "Plan 模式下 write_file 到计划文件允许",
        state5,
        expected_plan_mode=True,
        expected_plan_file_path=plan_path,
    ))

    # ---------- Test 6: 不在 Plan 模式下约束不生效 ----------
    state6 = build_state(
        [MockTC("write_file", {"path": "other.py", "content": "x"})],
        plan_mode=False,
        plan_file_path="",
    )
    results.append(run_test(
        "非 Plan 模式下 write_file 不受拦截",
        state6,
        expected_plan_mode=False,
        expected_plan_file_path="",
    ))

    # ---------- Test 7: exit_plan_mode - approve ----------
    def check_exit_approve(msgs):
        content = msgs[0].content
        if "计划已批准，已退出 Plan 模式" not in content:
            print(f"  FAIL: 未正确退出: {content[:200]!r}")
            return False
        return True

    with patch("builtins.input", return_value="a"):
        state7 = build_state(
            [MockTC("exit_plan_mode", {})],
            plan_mode=True,
            plan_file_path=plan_path,
        )
        results.append(run_test(
            "exit_plan_mode - 用户选择 approve",
            state7,
            expected_plan_mode=False,
            expected_plan_file_path="",
            check_callback=check_exit_approve,
        ))

    # ---------- Test 8: exit_plan_mode - reject (保持 Plan 模式) ----------
    def check_exit_reject(msgs):
        content = msgs[0].content
        if "用户拒绝了计划" not in content:
            print(f"  FAIL: 未返回拒绝消息: {content[:200]!r}")
            return False
        return True

    with patch("builtins.input", return_value="r"):
        state8 = build_state(
            [MockTC("exit_plan_mode", {})],
            plan_mode=True,
            plan_file_path=plan_path,
        )
        results.append(run_test(
            "exit_plan_mode - 用户选择 reject (保持 Plan)",
            state8,
            expected_plan_mode=True,
            expected_plan_file_path=plan_path,
            check_callback=check_exit_reject,
        ))

    # ---------- Test 9: exit_plan_mode - reject_and_exit ----------
    def check_exit_reject_exit(msgs):
        content = msgs[0].content
        if "用户拒绝了计划并退出 Plan 模式" not in content:
            print(f"  FAIL: 未正确退出: {content[:200]!r}")
            return False
        return True

    with patch("builtins.input", return_value="x"):
        state9 = build_state(
            [MockTC("exit_plan_mode", {})],
            plan_mode=True,
            plan_file_path=plan_path,
        )
        results.append(run_test(
            "exit_plan_mode - 用户选择 reject_and_exit",
            state9,
            expected_plan_mode=False,
            expected_plan_file_path="",
            check_callback=check_exit_reject_exit,
        ))

    # ---------- Test 10: exit_plan_mode - revise (保持 Plan) ----------
    def check_exit_revise(msgs):
        content = msgs[0].content
        if "用户要求修改计划" not in content:
            print(f"  FAIL: 未返回修改消息: {content[:200]!r}")
            return False
        return True

    with patch("builtins.input", return_value="v"):
        state10 = build_state(
            [MockTC("exit_plan_mode", {})],
            plan_mode=True,
            plan_file_path=plan_path,
        )
        results.append(run_test(
            "exit_plan_mode - 用户选择 revise (保持 Plan)",
            state10,
            expected_plan_mode=True,
            expected_plan_file_path=plan_path,
            check_callback=check_exit_revise,
        ))

    # ---------- Test 11: exit_plan_mode - 选择 option ----------
    def check_exit_option(msgs):
        content = msgs[0].content
        if "用户选择了方案 'Option A'" not in content:
            print(f"  FAIL: 未返回 option 消息: {content[:200]!r}")
            return False
        return True

    with patch("builtins.input", return_value="1"):
        state11 = build_state(
            [MockTC("exit_plan_mode", {
                "options": [
                    {"label": "Option A", "description": "Desc A"},
                    {"label": "Option B", "description": "Desc B"},
                ]
            })],
            plan_mode=True,
            plan_file_path=plan_path,
        )
        results.append(run_test(
            "exit_plan_mode - 用户选择 option",
            state11,
            expected_plan_mode=False,
            expected_plan_file_path="",
            check_callback=check_exit_option,
        ))

    # ---------- Test 12: exit_plan_mode 不在 Plan 模式下返回错误 ----------
    def check_exit_no_plan(msgs):
        content = msgs[0].content
        if "当前不在 Plan 模式中" not in content:
            print(f"  FAIL: 未返回错误: {content[:200]!r}")
            return False
        return True

    state12 = build_state(
        [MockTC("exit_plan_mode", {})],
        plan_mode=False,
        plan_file_path="",
    )
    results.append(run_test(
        "exit_plan_mode 不在 Plan 模式下返回错误",
        state12,
        expected_plan_mode=False,
        expected_plan_file_path="",
        check_callback=check_exit_no_plan,
    ))

    # ---------- 汇总 ----------
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
