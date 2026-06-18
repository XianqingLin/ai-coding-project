"""Benchmark 数据模型.

定义任务规格、评测方式、单次结果与聚合报告的 Schema。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationSpec:
    """客观判题配置."""

    type: str  # script | pytest | python
    command: str
    timeout: int = 120
    expected_exit_code: int = 0


@dataclass
class TaskSpec:
    """单个 benchmark 任务."""

    name: str
    description: str
    prompt: str
    evaluation: EvaluationSpec
    model: Optional[str] = None
    max_rounds: int = 5
    auto_approve: bool = True
    initial_files: List[str] = field(default_factory=list)
    source_dir: Optional[Path] = None

    @property
    def identifier(self) -> str:
        """返回安全的任务标识符."""
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in self.name)
        return safe or "task"


@dataclass
class TaskResult:
    """单个任务的运行结果."""

    name: str
    passed: bool
    status: str  # passed | failed | timeout | error | max_rounds_reached
    duration_seconds: float
    actual_rounds: int
    tool_calls_count: int
    message_count: int
    token_usage: int
    test_output: str = ""
    error: str = ""
    conversation: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "status": self.status,
            "duration_seconds": round(self.duration_seconds, 3),
            "actual_rounds": self.actual_rounds,
            "tool_calls_count": self.tool_calls_count,
            "message_count": self.message_count,
            "token_usage": self.token_usage,
            "test_output": self.test_output,
            "error": self.error,
            "conversation": self.conversation,
        }


@dataclass
class BenchmarkReport:
    """一次 benchmark 运行的完整报告."""

    metadata: Dict[str, Any]
    tasks: List[TaskResult]

    @property
    def summary(self) -> Dict[str, Any]:
        total = len(self.tasks)
        passed = sum(1 for t in self.tasks if t.passed)
        durations = [t.duration_seconds for t in self.tasks]
        rounds = [t.actual_rounds for t in self.tasks]
        tools = [t.tool_calls_count for t in self.tasks]
        tokens = [t.token_usage for t in self.tasks]

        def _avg(values: List[float]) -> float:
            return round(sum(values) / len(values), 3) if values else 0.0

        return {
            "total_tasks": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "avg_duration": _avg(durations),
            "avg_rounds": _avg(rounds),
            "avg_tool_calls": _avg(tools),
            "avg_token_usage": _avg(tokens),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "summary": self.summary,
            "tasks": [t.to_dict() for t in self.tasks],
        }
