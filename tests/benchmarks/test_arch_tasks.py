"""Architecture stress task 冒烟测试.

用 Mock LLM 验证任务配置本身是可解的，同时验证 runner 能正确处理
多文件编辑类任务。
"""

from pathlib import Path

from ai_coding.mock_llm import MockChatModel, mock_text, mock_tool_call
from benchmarks.loader import load_task
from benchmarks.runner import BenchmarkRunner


def _load_arch_task(name: str):
    task_dir = (
        Path(__file__).parent.parent.parent
        / "benchmarks"
        / "tasks"
        / "arch_stress"
        / name
    )
    assert task_dir.exists(), task_dir
    return load_task(task_dir)


def test_rename_function_task_solvable() -> None:
    task = _load_arch_task("rename_function")

    responses = [
        mock_tool_call(
            "write_file",
            {
                "path": "utils.py",
                "content": "def new_name(x):\n    return x * 2\n",
            },
            content="rename function in utils",
        ),
        mock_tool_call(
            "write_file",
            {
                "path": "main.py",
                "content": "from utils import new_name\n\n\ndef run():\n    return new_name(5)\n\n\nif __name__ == \"__main__\":\n    print(run())\n",
            },
            content="update import in main",
        ),
        mock_text("Done."),
    ]

    runner = BenchmarkRunner(
        model="mock",
        verbose=False,
        llm_factory=lambda: MockChatModel(responses=responses),
    )
    result = runner.run_task(task)

    assert result.passed is True
    assert result.status == "passed"
    assert result.tool_calls_count >= 2


def test_fix_bug_task_solvable() -> None:
    task = _load_arch_task("fix_bug_iterative")

    responses = [
        mock_tool_call(
            "write_file",
            {
                "path": "calculator.py",
                "content": "def add(a, b):\n    return a + b\n",
            },
            content="fix add function",
        ),
        mock_text("Done."),
    ]

    runner = BenchmarkRunner(
        model="mock",
        verbose=False,
        llm_factory=lambda: MockChatModel(responses=responses),
    )
    result = runner.run_task(task)

    assert result.passed is True
    assert result.status == "passed"


def test_long_file_edit_task_solvable() -> None:
    task = _load_arch_task("long_file_edit")

    responses = [
        mock_tool_call(
            "edit_file",
            {
                "path": "data_processor.py",
                "old_string": "def process_data(data):\n    \"\"\"Return a list where each element is doubled.\"\"\"\n    # FIXME: currently returns unchanged data\n    return data",
                "new_string": "def process_data(data):\n    \"\"\"Return a list where each element is doubled.\"\"\"\n    return [x * 2 for x in data]",
            },
            content="fix process_data",
        ),
        mock_text("Done."),
    ]

    runner = BenchmarkRunner(
        model="mock",
        verbose=False,
        llm_factory=lambda: MockChatModel(responses=responses),
    )
    result = runner.run_task(task)

    assert result.passed is True
    assert result.status == "passed"
