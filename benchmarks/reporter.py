"""Benchmark 报告输出.

支持 JSON 文件保存与命令行 Markdown 表格打印。
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

from benchmarks.spec import BenchmarkReport


def _safe_print(text: str) -> None:
    """安全打印，避免 Windows GBK 终端遇到非 ASCII 字符崩溃."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoded = text.encode(sys.stdout.encoding or "utf-8", errors="replace")
        sys.stdout.buffer.write(encoded + b"\n")


def save_report(report: BenchmarkReport, output_path: Path) -> Path:
    """保存报告为 JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    return output_path


def default_report_path(output_dir: Path, model: str) -> Path:
    """生成默认报告文件名."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_model = "".join(c if c.isalnum() or c in "_-" else "_" for c in model)
    return Path(output_dir) / f"benchmark_{safe_model}_{timestamp}.json"


def print_summary(report: BenchmarkReport) -> None:
    """在终端打印摘要."""
    summary = report.summary
    _safe_print("\n" + "=" * 70)
    _safe_print("Benchmark Summary")
    _safe_print("=" * 70)
    _safe_print(f"Model: {report.metadata.get('model', 'unknown')}")
    _safe_print(f"Tasks: {summary['total_tasks']}")
    _safe_print(f"Passed: {summary['passed']} | Failed: {summary['failed']}")
    _safe_print(f"Pass Rate: {summary['pass_rate'] * 100:.1f}%")
    _safe_print(f"Avg Duration: {summary['avg_duration']:.1f}s")
    _safe_print(f"Avg Rounds: {summary['avg_rounds']:.2f}")
    _safe_print(f"Avg Tool Calls: {summary['avg_tool_calls']:.1f}")
    _safe_print(f"Avg Token Usage: {summary['avg_token_usage']:.0f}")
    _safe_print("=" * 70)


def print_tasks_table(report: BenchmarkReport) -> None:
    """打印每个任务的表格."""
    header = "| Task | Status | Duration(s) | Rounds | Tool Calls | Tokens |"
    sep = "| --- | --- | --- | --- | --- | --- |"
    _safe_print("\n" + header)
    _safe_print(sep)
    for t in report.tasks:
        status = "passed" if t.passed else t.status
        _safe_print(
            f"| {t.name} | {status} | {t.duration_seconds:.1f} | "
            f"{t.actual_rounds} | {t.tool_calls_count} | {t.token_usage} |"
        )
