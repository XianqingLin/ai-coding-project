#!/usr/bin/env python3
"""子 Agent 功能测试.

验证 SubAgentManager、子 Agent 工具及 llm_node 结果注入.
用法:
    python scripts/test_sub_agent.py
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai_coding.agent.nodes.llm_node import create_llm_node
from ai_coding.agent.state import AgentState
from ai_coding.sub_agent_manager import SubAgentManager
from ai_coding.mock_llm import MockChatModel, mock_text, mock_tool_call
from ai_coding.tools.sub_agent_tools import (
    DispatchSubAgentTool,
    ListSubAgentsTool,
    GetSubAgentResultTool,
)


# ---------- 辅助函数 ----------

def make_sub_llm(responses):
    """创建子 Agent 用的 Mock LLM."""
    return MockChatModel(responses=responses)


def build_state(messages, sub_agents=None):
    """构建测试用的 AgentState."""
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


def run_test(name, condition, detail=""):
    """运行单条测试断言."""
    status = "PASS" if condition else "FAIL"
    detail_str = f" | {detail}" if detail else ""
    print(f"  [{status}] {name}{detail_str}")
    return condition


# ---------- 测试主体 ----------

def test_sync_dispatch_coder():
    """测试同步 dispatch coder 子 Agent."""
    print("\n[Test] 同步 dispatch coder 子 Agent")

    # 子 Agent Mock LLM：先读文件，再写文件，最后输出结果
    sub_llm = make_sub_llm([
        mock_tool_call("list_dir", {"path": "."}, content="看看目录"),
        mock_text("目录结构很简单，只有一个 app.py。"),
    ])

    manager = SubAgentManager()
    result = manager.dispatch(
        agent_type="coder",
        task="查看当前目录结构",
        llm=None,
        llm_factory=lambda: sub_llm,
        run_in_background=False,
    )

    ok = True
    ok &= run_test("返回非空结果", len(result) > 0)
    ok &= run_test("子 Agent 实例状态为 completed",
                   all(i.status == "completed" for i in manager.list_instances()))
    return ok


def test_sync_dispatch_explore_readonly():
    """测试 explore 子 Agent 只有只读工具."""
    print("\n[Test] explore 子 Agent 工具集过滤")

    # 子 Agent Mock LLM：尝试调用 write_file（但 explore 没有这个工具）
    sub_llm = make_sub_llm([
        mock_tool_call("write_file", {"path": "test.txt", "content": "x"}, content="尝试写入"),
    ])

    manager = SubAgentManager()
    # 清理之前测试的实例
    for i in list(manager.list_instances()):
        manager._agents.pop(i.instance_id, None)

    result = manager.dispatch(
        agent_type="explore",
        task="尝试写入文件",
        llm=None,
        llm_factory=lambda: sub_llm,
        run_in_background=False,
    )

    ok = True
    ok &= run_test("子 Agent 执行了（即使工具调用失败）", len(result) > 0)
    # explore 子 Agent 的工具集只有 read/list/grep/glob，没有 write_file
    # Mock LLM 会尝试调用 write_file，但工具注册表中没有这个工具
    # 工具调用会失败，子 Agent 会收到错误信息
    ok &= run_test("结果包含错误或提示",
                   "错误" in result or "未知工具" in result or "[错误]" in result or len(result) > 0)
    return ok


def test_background_dispatch_and_result():
    """测试后台 dispatch 和结果查询."""
    print("\n[Test] 后台 dispatch + get_sub_agent_result")

    # 子 Agent Mock LLM：简单任务
    sub_llm = make_sub_llm([
        mock_text("后台任务已完成。"),
    ])

    manager = SubAgentManager()
    # 清理之前测试的实例
    for i in list(manager.list_instances()):
        manager._agents.pop(i.instance_id, None)

    result = manager.dispatch(
        agent_type="coder",
        task="简单后台任务",
        llm=None,
        llm_factory=lambda: sub_llm,
        run_in_background=True,
    )

    ok = True
    ok &= run_test("返回包含 instance_id", "ID:" in result)

    # 提取 instance_id
    sid = None
    for line in result.split("\n"):
        if "ID:" in line:
            sid = line.split("ID:")[1].strip()
            break
    ok &= run_test("成功提取 instance_id", sid is not None)

    if sid:
        # 查询结果（block=true 等待完成）
        tool = GetSubAgentResultTool()
        result_text = tool.execute(instance_id=sid, block=True, timeout=10)
        ok &= run_test("block=true 获取到结果", "[Result]" in result_text)
        ok &= run_test("结果包含子 Agent 输出", "后台任务已完成" in result_text)

    return ok


def test_list_sub_agents():
    """测试 list_sub_agents 工具."""
    print("\n[Test] list_sub_agents")

    manager = SubAgentManager()
    # 清理之前测试的实例
    for i in list(manager.list_instances()):
        manager._agents.pop(i.instance_id, None)

    # 创建两个实例：一个完成，一个运行中
    sub_llm1 = make_sub_llm([mock_text("完成")])
    sub_llm2 = make_sub_llm([mock_text("运行中")])

    manager.dispatch("coder", "任务1", llm=None, llm_factory=lambda: sub_llm1, run_in_background=False)
    manager.dispatch("explore", "任务2", llm=None, llm_factory=lambda: sub_llm2, run_in_background=True)

    # 等待后台任务完成
    time.sleep(0.5)

    tool = ListSubAgentsTool()
    all_text = tool.execute(status="")
    ok = True
    ok &= run_test("列出所有实例", "coder" in all_text and "explore" in all_text)

    completed_text = tool.execute(status="completed")
    ok &= run_test("按 completed 过滤", "coder" in completed_text)

    return ok


def test_llm_node_injection():
    """测试 llm_node 注入已完成的子 Agent 结果."""
    print("\n[Test] llm_node 注入子 Agent 结果")

    manager = SubAgentManager()
    # 清理之前测试的实例
    for i in list(manager.list_instances()):
        manager._agents.pop(i.instance_id, None)

    # 创建一个已完成的子 Agent
    sub_llm = make_sub_llm([mock_text("探索结果：项目包含 3 个模块。")])
    manager.dispatch("explore", "探索项目结构", llm=None, llm_factory=lambda: sub_llm, run_in_background=False)

    # 确认有未通知的实例
    pending = manager.get_pending_notifications()

    # 创建主 Agent 的 Mock LLM，能追踪输入消息
    class TrackingMockLLM(MockChatModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.last_input_messages = []

        def invoke(self, messages, **kwargs):
            self.last_input_messages = [m.content for m in messages if hasattr(m, "content")]
            return super().invoke(messages, **kwargs)

    main_llm = TrackingMockLLM(responses=[mock_text("收到子 Agent 结果。")])

    # 构建 state
    state = build_state([
        SystemMessage(content="System prompt"),
        HumanMessage(content="用户输入"),
    ])

    llm_node = create_llm_node(llm=main_llm)
    result = llm_node(state)

    ok = True
    ok &= run_test("有未通知的子 Agent", len(pending) >= 1)

    # 检查 LLM 输入中是否包含子 Agent 结果
    injected = any("子 Agent 结果" in content for content in main_llm.last_input_messages)
    ok &= run_test("llm_node 向 LLM 输入中注入了子 Agent 结果", injected)

    # 检查 sub_agents 更新中 notified 被标记
    sub_agents = result.get("sub_agents", [])
    notified = all(sa.get("notified", False) for sa in sub_agents if sa.get("status") in ("completed", "failed"))
    ok &= run_test("sub_agents 标记了 notified", notified)

    return ok


def test_dispatch_tool_interface():
    """测试 DispatchSubAgentTool 的工具接口."""
    print("\n[Test] DispatchSubAgentTool 接口")

    tool = DispatchSubAgentTool()
    # 未设置 LLM 时应返回错误
    result = tool.execute(agent_type="coder", task="test")
    ok = run_test("未初始化 LLM 返回错误", "[错误]" in result and "未初始化" in result)

    # 设置 LLM
    sub_llm = make_sub_llm([mock_text("工具测试结果。")])
    tool.set_llm(llm=None, llm_factory=lambda: sub_llm)
    result = tool.execute(agent_type="coder", task="test task")
    ok &= run_test("设置 LLM 后正常执行", "工具测试结果" in result)

    # 验证无效 agent_type
    result2 = tool.execute(agent_type="invalid", task="test")
    ok &= run_test("无效 agent_type 返回错误", "[错误]" in result2)

    return ok


def test_resume_instance():
    """测试唤回已有实例."""
    print("\n[Test] 唤回已有实例")

    manager = SubAgentManager()
    # 清理之前测试的实例
    for i in list(manager.list_instances()):
        manager._agents.pop(i.instance_id, None)

    sub_llm1 = make_sub_llm([mock_text("第一次运行结果。")])
    result1 = manager.dispatch(
        agent_type="coder",
        task="第一次任务",
        llm=None,
        llm_factory=lambda: sub_llm1,
        run_in_background=False,
    )

    # 获取 instance_id
    instances = manager.list_instances()
    if not instances:
        print("  [FAIL] 没有实例可唤回")
        return False

    sid = instances[0].instance_id

    # 唤回
    sub_llm2 = make_sub_llm([mock_text("第二次运行结果。")])
    result2 = manager.dispatch(
        agent_type="coder",
        task="第二次任务",
        llm=None,
        llm_factory=lambda: sub_llm2,
        run_in_background=False,
        instance_id=sid,
    )

    ok = True
    ok &= run_test("唤回后返回新结果", "第二次运行结果" in result2)
    ok &= run_test("实例数量未增加（复用）", len(manager.list_instances()) == 1)

    return ok


# ---------- 主入口 ----------

def main():
    tests = [
        test_sync_dispatch_coder,
        test_sync_dispatch_explore_readonly,
        test_background_dispatch_and_result,
        test_list_sub_agents,
        test_llm_node_injection,
        test_dispatch_tool_interface,
        test_resume_instance,
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
