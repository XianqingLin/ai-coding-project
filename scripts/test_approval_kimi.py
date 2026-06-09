#!/usr/bin/env python3
"""真实 LLM 交互式授权体验测试.

用预设的 stdin 输入序列模拟用户在 approval_gate 处的选择，
观察 Kimi 在授权拦截下的完整行为.
"""

import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from ai_coding.agent.core import LangGraphAgent
from ai_coding.llm import create_lc_llm
from ai_coding.logger import setup_logging
from ai_coding.tools import DEFAULT_TOOLS
from agent_common import verify_test_task





def main():
    setup_logging(level="INFO")

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    task_name = "simple-calculator-bug"
    task_dir = PROJECT_ROOT / "test-tasks" / task_name
    instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")

    work_dir = Path(tempfile.mkdtemp(prefix=f"test-approval-{task_name}-"))
    repo_dir = work_dir / "repo"
    shutil.copytree(task_dir, repo_dir, ignore=shutil.ignore_patterns("tests"))

    original_dir = os.getcwd()
    os.chdir(repo_dir)

    # 模拟用户输入序列：
    # y = 同意这次 edit_file
    # y = 同意这次 edit_file
    # a = 全局授权 execute_command
    # 多准备几个 y 以防 LLM 额外调用
    simulated_input = "y\ny\na\ny\ny\ny\n"
    original_stdin = sys.stdin
    sys.stdin = io.StringIO(simulated_input)

    try:
        llm = create_lc_llm("kimi")
        agent = LangGraphAgent(
            llm=llm,
            tools=DEFAULT_TOOLS,
            max_iterations=15,
            streaming=False,
            auto_approve=False,  # 关键：关闭自动授权
        )

        print("=" * 60)
        print(f"[Task] {task_name}")
        print(f"[Mode] Interactive approval (simulated stdin)")
        print(f"[Simulated choices] y -> y -> a -> y -> y -> y")
        print("=" * 60)

        result = agent.run(instruction)

        print("\n" + "=" * 60)
        print("[Agent Reply]")
        print(result)
        print("=" * 60)

        # 验证
        verify_result = verify_test_task(task_dir, repo_dir, work_dir)
        print("\n[Verification]")
        if verify_result["reward"] == 1:
            print("[PASS] Validation passed")
        else:
            print(f"[FAIL] {verify_result['reason']}")

        # 会话统计
        print("\n[Session Stats]")
        print(f"  Messages: {len(agent.get_history())}")
        print(f"  Globally approved tools: {agent.state.get('globally_approved_tools', []) if agent.state else []}")

    finally:
        sys.stdin = original_stdin
        os.chdir(original_dir)


if __name__ == "__main__":
    sys.exit(main())
