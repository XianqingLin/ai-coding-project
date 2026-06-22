"""Benchmark CLI 入口.

用法:
    python -m benchmarks --tasks benchmarks/tasks --model kimi
    python -m benchmarks --tasks benchmarks/tasks/arch_stress/rename_function --model mock
"""

import argparse
import sys
from pathlib import Path

from benchmarks.loader import load_task, load_tasks
from benchmarks.reporter import (
    default_report_path,
    print_summary,
    print_tasks_table,
    save_report,
)
from benchmarks.runner import BenchmarkRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AI Coding 可量化 Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python -m benchmarks --tasks benchmarks/tasks --model kimi
  python -m benchmarks --tasks benchmarks/tasks/arch_stress/rename_function --model mock
""",
    )
    parser.add_argument(
        "--tasks",
        required=True,
        help="任务目录或单个任务目录，包含 task.json",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="覆盖默认 LLM 提供商，如 kimi / openai / mock",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="报告输出路径（JSON），默认保存到 benchmarks/reports/",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="保留临时工作目录，便于调试失败任务",
    )
    parser.add_argument(
        "--capture-conversation",
        action="store_true",
        help="在报告中保存完整对话/事件流",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="只输出最终摘要",
    )
    args = parser.parse_args()

    tasks_path = Path(args.tasks).resolve()
    if tasks_path.is_dir() and (tasks_path / "task.json").exists():
        tasks = [load_task(tasks_path)]
    else:
        tasks = load_tasks(tasks_path)

    if not tasks:
        print(f"未找到任务: {tasks_path}", file=sys.stderr)
        return 1

    runner = BenchmarkRunner(
        model=args.model,
        keep_work_dir=args.keep_work_dir,
        capture_conversation=args.capture_conversation,
        verbose=not args.quiet,
    )
    report = runner.run_all(tasks)

    print_summary(report)
    if not args.quiet:
        print_tasks_table(report)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = default_report_path(
            Path(__file__).parent / "reports",
            report.metadata.get("model", "unknown"),
        )
    saved = save_report(report, output_path)
    print(f"\nReport saved: {saved}")
    return 0 if report.summary["pass_rate"] == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
