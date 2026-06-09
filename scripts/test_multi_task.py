#!/usr/bin/env python3
"""多轮对话测试运行器.

在一个 Agent 会话下依次完成多个任务，验证上下文保持能力.

用法:
    # 使用 mock 完成2个任务
    python scripts/test_multi_task.py --tasks simple-calculator-bug simple-config-validation

    # 使用真实 LLM 完成3个任务
    python scripts/test_multi_task.py --tasks simple-calculator-bug simple-config-validation multi-method-implementation --provider kimi
"""

import argparse
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from ai_coding.agent.core import LangGraphAgent
from ai_coding.llm import create_lc_llm
from ai_coding.logger import setup_logging
from ai_coding.mock_llm import MockChatModel
from ai_coding.tools import DEFAULT_TOOLS
from agent_common import verify_test_task





def main():
    parser = argparse.ArgumentParser(
        description="多轮对话测试运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/test_multi_task.py --tasks simple-calculator-bug simple-config-validation
  python scripts/test_multi_task.py --tasks simple-calculator-bug simple-config-validation multi-method-implementation --provider kimi
        """,
    )
    parser.add_argument("--tasks", nargs="+", required=True, help="任务名称列表")
    parser.add_argument("--tasks-dir", default="test-tasks", help="任务目录（默认 test-tasks）")
    parser.add_argument("--provider", default="mock", help="LLM provider（默认 mock）")
    parser.add_argument("--max-iters", type=int, default=15, help="最大迭代次数")
    parser.add_argument("--clear-todos", action="store_true", help="任务间清空 todo 列表")
    parser.add_argument("--keep-snapshots", action="store_true", help="任务间保留 file_snapshots（可能混乱）")
    args = parser.parse_args()

    setup_logging(level="INFO")

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # 创建 LLM
    if args.provider == "mock":
        from test_task import BUILTIN_MOCK_RESPONSES
        all_responses = []
        for task_name in args.tasks:
            responses = BUILTIN_MOCK_RESPONSES.get(task_name, [])
            if not responses:
                print(f"[Warn] 任务 '{task_name}' 没有内置 Mock 响应")
            all_responses.extend(responses)
        print(f"[Mock] 总响应数: {len(all_responses)}")
        llm = MockChatModel(responses=all_responses)
    else:
        llm = create_lc_llm(args.provider)

    work_dir = Path(tempfile.mkdtemp(prefix="test-multi-task-"))
    print(f"[Work] 工作目录: {work_dir}")

    agent = LangGraphAgent(
        llm=llm,
        tools=DEFAULT_TOOLS,
        max_iterations=args.max_iters,
        streaming=False,
        auto_approve=True,
    )

    original_dir = os.getcwd()
    results = []

    for i, task_name in enumerate(args.tasks, 1):
        task_dir = PROJECT_ROOT / args.tasks_dir / task_name
        if not task_dir.exists():
            print(f"❌ 错误: 任务目录不存在: {task_dir}")
            results.append({"task": task_name, "reward": 0, "reason": "task_not_found"})
            continue

        instruction_path = task_dir / "instruction.md"
        if not instruction_path.exists():
            print(f"❌ 错误: 指令文件不存在: {instruction_path}")
            results.append({"task": task_name, "reward": 0, "reason": "no_instruction"})
            continue

        instruction = instruction_path.read_text(encoding="utf-8")

        task_work_dir = work_dir / f"task-{i}-{task_name}"
        repo_dir = task_work_dir / "repo"
        shutil.copytree(task_dir, repo_dir, ignore=shutil.ignore_patterns("tests"))

        os.chdir(repo_dir)

        # 任务间状态管理
        if i > 1 and agent.state:
            if not args.keep_snapshots:
                print(f"[State] 清空 file_snapshots")
                agent.state["file_snapshots"] = {}
            if args.clear_todos:
                print(f"[State] 清空 todos")
                agent.state["todos"] = []

        # 任务标记前缀（帮助 LLM 识别任务切换）
        marked_instruction = f"【任务 {i}/{len(args.tasks)}: {task_name}】\n\n{instruction}"

        print(f"\n{'='*60}")
        print(f"[Task {i}/{len(args.tasks)}] {task_name}")
        print(f"[Instruction] {instruction[:120]}...")
        print(f"{'='*60}")

        try:
            result = agent.run(marked_instruction)
        except Exception as e:
            print(f"[Error] {e}")
            traceback.print_exc()
            results.append({"task": task_name, "reward": 0, "reason": f"agent_error: {e}"})
            continue

        print(f"\n[Agent Reply]")
        print(result[:500])
        if len(result) > 500:
            print(f"... ({len(result)} chars total)")

        print(f"\n{'='*60}")
        print("[Verification]")
        verify_result = verify_test_task(task_dir, repo_dir, task_work_dir)

        reward = verify_result.get("reward", "N/A")
        if reward == 1:
            print("[PASS] 验证通过")
        elif reward == 0:
            reason = verify_result.get("reason", "unknown")
            print(f"[FAIL] 验证失败 (reason={reason})")
            if verify_result.get("stderr"):
                print(f"  错误: {verify_result['stderr'][:300]}")
        else:
            print(f"[?] 验证结果: reward={reward}")

        results.append({
            "task": task_name,
            "reward": reward,
            "reason": verify_result.get("reason"),
        })

    os.chdir(original_dir)

    # 汇总
    print(f"\n{'='*60}")
    print("[Summary]")
    passed = sum(1 for r in results if r["reward"] == 1)
    total = len(results)
    print(f"  通过: {passed}/{total}")
    for r in results:
        status = "✅" if r["reward"] == 1 else "❌"
        print(f"  {status} {r['task']} (reason={r.get('reason', 'N/A')})")

    # 会话统计
    print(f"\n[Session Stats]")
    history = agent.get_history()
    print(f"  总消息数: {len(history)}")
    if agent.state:
        print(f"  待办数: {len(agent.state.get('todos', []))}")
        print(f"  文件快照数: {len(agent.state.get('file_snapshots', {}))}")
    else:
        print(f"  状态: 未初始化")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
