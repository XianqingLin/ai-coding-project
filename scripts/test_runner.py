#!/usr/bin/env python3
"""任务测试运行器.

加载 test-tasks 中的任务，在隔离的临时工作目录中运行 Agent，
支持持久化验证和交互式观察.

用法:
    # 运行单个任务（自动批准工具）
    python scripts/test_runner.py --task implement-api-pagination --auto-approve

    # 运行并验证持久化（模拟重启恢复）
    python scripts/test_runner.py --task implement-api-pagination --auto-approve --persist-check

    # 限制迭代次数，快速验证
    python scripts/test_runner.py --task simple-calculator-bug --auto-approve --max-iters 3

    # 列出所有可用任务
    python scripts/test_runner.py --list
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_coding.config import DEFAULT_LLM_PROVIDER
from ai_coding.llm import create_lc_llm
from ai_coding.logger import setup_logging, get_logger
# 先导入 agent 包（避免 tools -> collaboration_tools -> agent 循环导入）
from ai_coding.agent import SessionManager
from ai_coding.tools import DEFAULT_TOOLS
from ai_coding.persistence import StorageEngine

logger = get_logger(__name__)

# test-tasks 搜索路径（支持项目内或同级目录）
_TASKS_SEARCH_PATHS = [
    PROJECT_ROOT / "test-tasks",
    PROJECT_ROOT.parent / "ai-coding-test-tasks" / "test-tasks",
    PROJECT_ROOT.parent / "test-tasks",
]


def _find_tasks_dir() -> Path:
    """定位 test-tasks 目录."""
    for p in _TASKS_SEARCH_PATHS:
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError("找不到 test-tasks 目录，请确保任务文件夹存在")


def _list_tasks() -> list:
    """列出所有可用任务."""
    tasks_dir = _find_tasks_dir()
    tasks = []
    for d in sorted(tasks_dir.iterdir()):
        if d.is_dir() and (d / "instruction.md").exists():
            tasks.append(d.name)
    return tasks


def _load_task(task_name: str) -> tuple:
    """加载任务指令和源目录.

    Returns:
        (instruction_text, task_source_dir)

    """
    tasks_dir = _find_tasks_dir()
    task_dir = tasks_dir / task_name
    if not task_dir.exists():
        raise FileNotFoundError(f"任务不存在: {task_name}")
    instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
    return instruction, task_dir


def _prepare_work_dir(task_dir: Path) -> Path:
    """将任务代码复制到临时工作目录."""
    work_dir = Path(tempfile.mkdtemp(prefix=f"ai-coding-test-{task_dir.name}-"))
    # 复制所有文件（排除 .git）
    for item in task_dir.iterdir():
        if item.name == ".git":
            continue
        dest = work_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    return work_dir


def _print_banner(task_name: str, work_dir: Path, provider: str) -> None:
    """打印测试横幅."""
    print("=" * 60)
    print(f"  AI-Coding 任务测试")
    print("=" * 60)
    print(f"  任务: {task_name}")
    print(f"  工作目录: {work_dir}")
    print(f"  模型: {provider}")
    print("=" * 60)


def _run_task(
    task_name: str,
    auto_approve: bool = False,
    max_iters: int = 10,
    persist_check: bool = False,
    provider: str = DEFAULT_LLM_PROVIDER,
) -> dict:
    """运行单个任务并返回结果.

    Returns:
        {"success": bool, "work_dir": Path, "session_id": str, "elapsed": float}

    """
    instruction, task_dir = _load_task(task_name)
    work_dir = _prepare_work_dir(task_dir)

    _print_banner(task_name, work_dir, provider)

    # 保存原始 cwd，运行结束后恢复
    original_cwd = os.getcwd()
    os.chdir(work_dir)

    try:
        llm = create_lc_llm(provider)

        # 创建 SessionManager
        sm = SessionManager(
            llm_factory=lambda: create_lc_llm(provider),
            tools=DEFAULT_TOOLS,
            auto_approve=auto_approve,
            work_dir=str(work_dir),
        )

        session = sm.current
        session_id = session.session_id if session else ""
        print(f"\n[会话] {session.name} ({session_id}) 已创建\n")

        # 运行任务
        start = time.time()
        agent = sm.get_current_agent()
        if agent:
            agent.max_iterations = max_iters
            print("-" * 60)
            print(f"[任务指令]\n{instruction}\n")
            print("-" * 60)
            print("[Agent 开始执行...]\n")
            result = agent.run(instruction)
            print(f"\n[Agent 结果]\n{result}\n")
        else:
            print("[错误] 无法获取当前 Agent")
            return {"success": False, "work_dir": work_dir, "session_id": session_id, "elapsed": 0}

        elapsed = time.time() - start

        # 持久化验证
        if persist_check:
            print("-" * 60)
            print("[持久化验证]")
            storage = StorageEngine()
            state_text = storage.load_state(str(work_dir), session_id)
            meta = storage.load_meta(str(work_dir), session_id)
            wire = storage.load_wire(str(work_dir), session_id, "main")

            ok = True
            if state_text:
                print(f"  [OK] state.json 存在 ({len(state_text)} chars)")
            else:
                print(f"  [FAIL] state.json 不存在")
                ok = False

            if meta:
                print(f"  [OK] meta.json 存在 (thread_id={meta.get('thread_id')})")
            else:
                print(f"  [FAIL] meta.json 不存在")
                ok = False

            if wire:
                print(f"  [OK] wire.jsonl 存在 ({len(wire)} records)")
            else:
                print(f"  [WARN] wire.jsonl 为空或不存在")

            # 模拟重启：新建 SessionManager 验证恢复
            print("  模拟进程重启...")
            sm2 = SessionManager(
                llm_factory=lambda: create_lc_llm(provider),
                tools=DEFAULT_TOOLS,
                auto_approve=auto_approve,
                work_dir=str(work_dir),
            )
            recovered = sm2.current
            if recovered and recovered.agent.state is not None:
                msg_count = len(recovered.agent.get_history())
                print(f"  [OK] 恢复成功，消息数: {msg_count}")
            else:
                print(f"  [FAIL] 恢复失败，AgentState 为空")
                ok = False

            print(f"\n  持久化验证: {'PASS' if ok else 'FAIL'}")

        return {"success": True, "work_dir": work_dir, "session_id": session_id, "elapsed": elapsed}

    except Exception as e:
        logger.error(f"任务执行失败: {e}", exc_info=True)
        print(f"\n[错误] {e}\n")
        return {"success": False, "work_dir": work_dir, "session_id": "", "elapsed": 0}

    finally:
        os.chdir(original_cwd)


def _ensure_utf8() -> None:
    """Windows 终端兼容：强制 UTF-8 输出."""
    import sys
    if sys.platform == "win32":
        import io
        if sys.stdout.encoding != "utf-8":
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if sys.stderr.encoding != "utf-8":
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def main() -> int:
    _ensure_utf8()
    parser = argparse.ArgumentParser(description="AI-Coding 任务测试运行器")
    parser.add_argument("--task", default="", help="要运行的任务名")
    parser.add_argument("--list", action="store_true", help="列出所有可用任务")
    parser.add_argument("--auto-approve", action="store_true", help="自动批准工具调用")
    parser.add_argument("--max-iters", type=int, default=10, help="最大迭代次数")
    parser.add_argument("--persist-check", action="store_true", help="验证持久化功能")
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER, help="LLM 提供商")
    args = parser.parse_args()

    setup_logging()

    if args.list:
        tasks = _list_tasks()
        print(f"可用任务 ({len(tasks)}):")
        for i, t in enumerate(tasks, 1):
            print(f"  {i}. {t}")
        return 0

    if not args.task:
        tasks = _list_tasks()
        print("请指定任务名 (--task)，可用任务:")
        for i, t in enumerate(tasks, 1):
            print(f"  {i}. {t}")
        return 1

    result = _run_task(
        task_name=args.task,
        auto_approve=args.auto_approve,
        max_iters=args.max_iters,
        persist_check=args.persist_check,
        provider=args.provider,
    )

    print("=" * 60)
    if result["success"]:
        print(f"  Task completed | elapsed: {result['elapsed']:.1f}s")
    else:
        print("  Task failed")
    print(f"  Work dir: {result['work_dir']}")
    print("=" * 60)

    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
