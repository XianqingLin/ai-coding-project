#!/usr/bin/env python3
"""Test-tasks 轻量测试运行器.

用于在 test-tasks 目录下运行单个测试任务，验证 Agent 架构.

用法:
    # 使用内置 Mock 响应运行 simple-calculator-bug
    python scripts/test_task.py --task simple-calculator-bug

    # 使用真实 LLM 运行
    python scripts/test_task.py --task simple-calculator-bug --provider kimi
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
from ai_coding.mock_llm import MockChatModel, mock_tool_call, mock_text
from ai_coding.tools import DEFAULT_TOOLS
from eval_deepswe import load_instruction
from agent_common import verify_test_task


# 内置的 Mock 响应序列（按任务名）
BUILTIN_MOCK_RESPONSES = {
    "simple-calculator-bug": [
        mock_tool_call("list_dir", {"path": "."}, content="看看目录结构"),
        mock_tool_call("read_file", {"path": "calculator.py"}, content="读取代码文件"),
        mock_tool_call(
            "edit_file",
            {
                "path": "calculator.py",
                "old_string": "# TODO: implement factorial\ndef factorial(n):\n    pass",
                "new_string": "def factorial(n):\n    if n == 0:\n        return 1\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result",
            },
            content="实现阶乘函数",
        ),
        mock_tool_call(
            "edit_file",
            {
                "path": "calculator.py",
                "old_string": "def divide(a, b):\n    return a / b",
                "new_string": 'def divide(a, b):\n    if b == 0:\n        raise ZeroDivisionError("division by zero")\n    return a / b',
            },
            content="添加除零检查",
        ),
        mock_tool_call(
            "execute_command",
            {"command": 'python -c "from calculator import factorial, divide; print(factorial(5), divide(10, 2)); divide(1, 0)"'},
            content="验证修复结果",
        ),
        mock_text("任务完成！已修复 calculator.py 中的两个 bug：\n1. 实现了 factorial 函数\n2. 为 divide 函数添加了除零检查"),
    ],
    "simple-config-validation": [
        mock_tool_call("list_dir", {"path": "."}, content="看看目录结构"),
        mock_tool_call("read_file", {"path": "validator.py"}, content="读取验证器代码"),
        mock_tool_call("read_file", {"path": "config.py"}, content="读取配置"),
        mock_tool_call("read_file", {"path": "app.py"}, content="读取应用代码"),
        mock_tool_call(
            "edit_file",
            {
                "path": "config.py",
                "old_string": 'APP_NAME = "MyApp"\nMAX_RETRIES = 3',
                "new_string": 'APP_NAME = "MyApp"\nMAX_RETRIES = 3\nTIMEOUT = 30',
            },
            content="添加 TIMEOUT 配置",
        ),
        mock_tool_call(
            "edit_file",
            {
                "path": "validator.py",
                "old_string": "def validate_config(config):\n    # TODO: implement validation\n    pass",
                "new_string": 'def validate_config(config):\n    required_keys = ["name", "retries", "timeout"]\n    for key in required_keys:\n        if key not in config:\n            raise ValueError(f"Missing required key: {key}")\n    if not isinstance(config["name"], str) or not config["name"]:\n        raise ValueError("name must be a non-empty string")\n    if not isinstance(config["retries"], int) or not (0 <= config["retries"] <= 10):\n        raise ValueError("retries must be an integer between 0 and 10")\n    if not isinstance(config["timeout"], int) or not (1 <= config["timeout"] <= 300):\n        raise ValueError("timeout must be an integer between 1 and 300")',
            },
            content="实现配置验证逻辑",
        ),
        mock_tool_call(
            "edit_file",
            {
                "path": "app.py",
                "old_string": "from config import APP_NAME, MAX_RETRIES\nfrom validator import validate_config\n\n\ndef run_app():\n    config = {\"name\": APP_NAME, \"retries\": MAX_RETRIES}\n    validate_config(config)\n    print(f\"Running {APP_NAME} with {MAX_RETRIES} retries\")",
                "new_string": "from config import APP_NAME, MAX_RETRIES, TIMEOUT\nfrom validator import validate_config\n\n\ndef run_app():\n    config = {\"name\": APP_NAME, \"retries\": MAX_RETRIES, \"timeout\": TIMEOUT}\n    validate_config(config)\n    print(f\"Running {APP_NAME} with {MAX_RETRIES} retries\")",
            },
            content="导入 TIMEOUT 并加入 config 字典",
        ),
        mock_tool_call(
            "execute_command",
            {"command": "python test_validation.py"},
            content="运行测试验证",
        ),
        mock_text("任务完成！已修复并增强 simple-config-validation 的配置验证功能。"),
    ],
}


def main():
    parser = argparse.ArgumentParser(
        description="Test-tasks 轻量测试运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/test_task.py --task simple-calculator-bug
  python scripts/test_task.py --task simple-calculator-bug --provider mock
  python scripts/test_task.py --task simple-config-validation --provider kimi
        """,
    )
    parser.add_argument("--task", required=True, help="任务名称（如 simple-calculator-bug）")
    parser.add_argument("--tasks-dir", default="test-tasks", help="任务目录（默认 test-tasks）")
    parser.add_argument("--provider", default="mock", help="LLM provider（默认 mock）")
    parser.add_argument("--max-iters", type=int, default=15, help="最大迭代次数")
    parser.add_argument("--timeout", type=int, default=120, help="Agent 超时时间（秒）")
    args = parser.parse_args()

    setup_logging(level="INFO")

    # 修复 Windows 控制台编码问题
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    task_dir = PROJECT_ROOT / args.tasks_dir / args.task
    if not task_dir.exists():
        print(f"❌ 错误: 任务目录不存在: {task_dir}")
        return 1

    instruction_path = task_dir / "instruction.md"
    if not instruction_path.exists():
        print(f"❌ 错误: 指令文件不存在: {instruction_path}")
        return 1

    instruction = instruction_path.read_text(encoding="utf-8")

    # 创建 LLM
    if args.provider == "mock":
        responses = BUILTIN_MOCK_RESPONSES.get(args.task)
        if responses:
            print(f"[Mock] 使用内置 Mock 响应序列（{len(responses)} 条）")
            llm = MockChatModel(responses=responses)
        else:
            print(f"[Warn] 任务 '{args.task}' 没有内置 Mock 响应，使用空 Mock")
            llm = MockChatModel()
    else:
        llm = create_lc_llm(args.provider)

    # 创建临时工作目录（避免污染原始任务目录）
    work_dir = Path(tempfile.mkdtemp(prefix=f"test-task-{args.task}-"))
    print(f"[Work] 工作目录: {work_dir}")

    # 复制任务目录到工作目录作为 repo（排除 tests 目录，避免 LLM 读到测试补丁）
    repo_dir = work_dir / "repo"
    shutil.copytree(task_dir, repo_dir, ignore=shutil.ignore_patterns("tests"))

    original_dir = os.getcwd()
    os.chdir(repo_dir)

    try:
        agent = LangGraphAgent(
            llm=llm,
            tools=DEFAULT_TOOLS,
            max_iterations=args.max_iters,
            streaming=False,
            auto_approve=True,
        )

        print(f"\n{'='*60}")
        print(f"[Task] {args.task}")
        print(f"[Provider] {args.provider}")
        print(f"[Instruction] {instruction[:100]}...")
        print(f"{'='*60}")

        result = agent.run(instruction)

        print(f"\n{'='*60}")
        print("[Agent Reply]")
        print(result)
        print(f"{'='*60}")

        print(f"\n[History Stats]")
        history = agent.get_history()
        print(f"  消息数: {len(history)}")
        for i, msg in enumerate(history):
            role = msg.get("role", "unknown")
            print(f"  [{i}] {role}: {msg.get('content', '')[:70]!r}")

        # 验证
        print(f"\n{'='*60}")
        print("[Verification]")
        verify_result = verify_test_task(task_dir, repo_dir, work_dir)

        reward = verify_result.get("reward", "N/A")
        if reward == 1:
            print("[PASS] 验证通过")
        elif reward == 0:
            reason = verify_result.get("reason", "unknown")
            print(f"[FAIL] 验证失败 (reason={reason})")
        else:
            print(f"[?] 验证结果: reward={reward}")

        if verify_result.get("output"):
            print(f"\n  输出:\n{verify_result['output']}")
        if verify_result.get("stderr"):
            print(f"\n  错误:\n{verify_result['stderr']}")

    except Exception as e:
        print(f"[Error] {e}")
        traceback.print_exc()
        return 1

    finally:
        os.chdir(original_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
