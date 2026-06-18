"""Benchmark runner 单元测试.

使用 Mock LLM 驱动一个最小任务，验证 runner 的判题、指标收集与报告生成。
"""

import json
from pathlib import Path

import pytest

from ai_coding.mock_llm import MockChatModel, mock_text, mock_tool_call
from benchmarks.loader import load_task, load_tasks
from benchmarks.runner import BenchmarkRunner


SOLUTION_TEST_SCRIPT = '''\
import py_compile
import sys

sys.path.insert(0, ".")

try:
    py_compile.compile("solution.py", doraise=True)
except Exception as e:
    print(f"[FAIL] syntax: {e}")
    sys.exit(1)

import solution
assert solution.answer == 42, "answer should be 42"
print("[PASS] solution ok")
'''


def _make_minimal_task(tmp_path: Path, prompt: str) -> Path:
    """创建一个最小 benchmark 任务目录."""
    task_dir = tmp_path / "minimal_task"
    task_dir.mkdir()
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_solution.py").write_text(SOLUTION_TEST_SCRIPT, encoding="utf-8")

    task_data = {
        "name": "minimal_task",
        "description": "Write a tiny solution module",
        "prompt": prompt,
        "max_rounds": 3,
        "auto_approve": True,
        "initial_files": ["tests/test_solution.py"],
        "evaluation": {
            "type": "script",
            "command": "python tests/test_solution.py",
            "timeout": 30,
            "expected_exit_code": 0,
        },
    }
    (task_dir / "task.json").write_text(json.dumps(task_data), encoding="utf-8")
    return task_dir


def test_load_task_and_runner_pass(tmp_path: Path) -> None:
    task_dir = _make_minimal_task(tmp_path, "Create solution.py with answer=42.")
    task = load_task(task_dir)

    responses = [
        mock_tool_call(
            "write_file",
            {"path": "solution.py", "content": "answer = 42\n"},
            content="write solution",
        ),
        mock_text("Task complete."),
    ]
    runner = BenchmarkRunner(
        model="mock",
        verbose=False,
        llm_factory=lambda: MockChatModel(responses=responses),
    )
    result = runner.run_task(task)

    assert result.passed is True
    assert result.status == "passed"
    assert result.actual_rounds == 1
    assert result.tool_calls_count >= 1
    assert result.message_count > 0
    assert "[PASS] solution ok" in result.test_output


def test_runner_reports_failure(tmp_path: Path) -> None:
    task_dir = _make_minimal_task(tmp_path, "Create solution.py with answer=42.")
    task = load_task(task_dir)

    # 写入错误内容，导致测试失败
    responses = [
        mock_tool_call(
            "write_file",
            {"path": "solution.py", "content": "answer = 7\n"},
            content="write wrong solution",
        ),
        mock_text("Done."),
    ]
    runner = BenchmarkRunner(
        model="mock",
        verbose=False,
        llm_factory=lambda: MockChatModel(responses=responses),
    )
    result = runner.run_task(task)

    assert result.passed is False
    assert result.status in ("failed", "max_rounds_reached")
    assert "answer should be 42" in result.test_output


def test_runner_multi_round_feedback(tmp_path: Path) -> None:
    task_dir = _make_minimal_task(tmp_path, "Create solution.py with answer=42.")
    task = load_task(task_dir)

    # 第一轮写错并结束；第二轮收到测试反馈后修正
    responses = [
        mock_tool_call(
            "write_file",
            {"path": "solution.py", "content": "answer = 7\n"},
            content="write wrong solution",
        ),
        mock_text("Done."),
        mock_tool_call(
            "write_file",
            {"path": "solution.py", "content": "answer = 42\n"},
            content="fix solution",
        ),
        mock_text("Fixed."),
    ]
    runner = BenchmarkRunner(
        model="mock",
        verbose=False,
        llm_factory=lambda: MockChatModel(responses=responses),
    )
    result = runner.run_task(task)

    assert result.passed is True
    assert result.status == "passed"
    assert result.actual_rounds == 2


def test_run_all_aggregates_summary(tmp_path: Path) -> None:
    task_dir = _make_minimal_task(tmp_path, "Create solution.py with answer=42.")
    task = load_task(task_dir)

    responses = [
        mock_tool_call(
            "write_file",
            {"path": "solution.py", "content": "answer = 42\n"},
            content="write solution",
        ),
        mock_text("Done."),
    ]
    runner = BenchmarkRunner(
        model="mock",
        verbose=False,
        llm_factory=lambda: MockChatModel(responses=responses),
    )
    report = runner.run_all([task])

    assert report.summary["total_tasks"] == 1
    assert report.summary["passed"] == 1
    assert report.summary["pass_rate"] == 1.0
    assert report.tasks[0].passed is True


def test_load_tasks_discovers_directory(tmp_path: Path) -> None:
    _make_minimal_task(tmp_path, "Create solution.py with answer=42.")
    tasks = load_tasks(tmp_path)
    assert len(tasks) == 1
    assert tasks[0].name == "minimal_task"
    assert tasks[0].source_dir is not None
