"""Benchmark 运行器.

在隔离的临时目录中实例化 AgentService，按任务配置多轮运行，
最后执行客观判题脚本并收集量化指标。
"""

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# 保证 benchmarks 包在任意位置被导入时都能找到 src 下的 ai_coding
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
_src_path = str(_PROJECT_ROOT / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from ai_coding.agent import AgentService
from ai_coding.agent.events import (
    AgentEvent,
    AssistantChunkEvent,
    AssistantEndEvent,
    ToolCallEvent,
    UserInputEvent,
)
from benchmarks.loader import copy_initial_files
from benchmarks.spec import BenchmarkReport, EvaluationSpec, TaskResult, TaskSpec


class BenchmarkRunner:
    """Benchmark 运行器."""

    def __init__(
        self,
        model: Optional[str] = None,
        keep_work_dir: bool = False,
        capture_conversation: bool = False,
        verbose: bool = True,
        llm_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.model = model
        self.keep_work_dir = keep_work_dir
        self.capture_conversation = capture_conversation
        self.verbose = verbose
        self.llm_factory = llm_factory

    def run_all(self, tasks: List[TaskSpec]) -> BenchmarkReport:
        """顺序运行所有任务并返回报告."""
        metadata: Dict[str, Any] = {
            "model": self.model or "default",
            "total_tasks": len(tasks),
        }
        results: List[TaskResult] = []
        for task in tasks:
            result = self.run_task(task)
            results.append(result)
            if self.verbose:
                self._print_task_result(result)
        return BenchmarkReport(metadata=metadata, tasks=results)

    def run_task(self, task: TaskSpec) -> TaskResult:
        """运行单个任务."""
        work_dir = Path(tempfile.mkdtemp(prefix=f"bench_{task.identifier}_")).resolve()
        try:
            if task.source_dir is not None:
                copy_initial_files(task.source_dir, work_dir, task.initial_files)
            return self._run_in_work_dir(task, work_dir)
        finally:
            if not self.keep_work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)

    def _run_in_work_dir(self, task: TaskSpec, work_dir: Path) -> TaskResult:
        start_time = time.time()
        service = AgentService(
            work_dir=str(work_dir),
            llm_provider=task.model or self.model,
            auto_approve=task.auto_approve,
            enable_env_info=True,
            llm_factory=self.llm_factory,
        )
        service.create_session(name=task.name)

        tool_calls_count = 0
        last_test_output = ""
        status = "max_rounds_reached"
        passed = False
        conversation: Optional[List[Dict[str, Any]]] = (
            [] if self.capture_conversation else None
        )
        error_message = ""

        for round_idx in range(1, task.max_rounds + 1):
            user_input = (
                task.prompt
                if round_idx == 1
                else self._build_feedback(last_test_output)
            )

            if conversation is not None:
                conversation.append({"role": "user", "content": user_input})

            assistant_text, calls = self._run_agent_round(
                service, user_input, conversation
            )
            tool_calls_count += calls

            try:
                exit_code, test_output = self._run_evaluation(task.evaluation, work_dir)
            except subprocess.TimeoutExpired as e:
                status = "timeout"
                error_message = f"判题脚本超时 ({task.evaluation.timeout}s)"
                last_test_output = self._truncate(
                    e.stdout.decode("utf-8", errors="replace") if e.stdout else "",
                    4000,
                )
                break
            except Exception as e:
                status = "error"
                error_message = f"判题脚本执行异常: {e}"
                break

            last_test_output = self._truncate(test_output, 4000)

            if exit_code == task.evaluation.expected_exit_code:
                passed = True
                status = "passed"
                break

            # 最后一轮仍未通过则标记 failed
            if round_idx == task.max_rounds:
                status = "failed"

        duration = time.time() - start_time
        context_usage = service.get_context_usage()
        stats = service.get_stats()

        return TaskResult(
            name=task.name,
            passed=passed,
            status=status,
            duration_seconds=duration,
            actual_rounds=(
                round_idx
                if passed or status in ("failed", "timeout")
                else task.max_rounds
            ),
            tool_calls_count=tool_calls_count,
            message_count=stats.get("message_count", 0),
            token_usage=context_usage.get("used_tokens", 0),
            test_output=last_test_output,
            error=error_message,
            conversation=conversation,
        )

    def _run_agent_round(
        self,
        service: AgentService,
        user_input: str,
        conversation: Optional[List[Dict[str, Any]]],
    ) -> tuple[str, int]:
        """发送一轮消息，返回 assistant 文本和工具调用次数."""
        tool_calls_count = 0
        assistant_text = ""
        for event in service.send_message_stream(user_input):
            event = self._normalize_event(event)
            if isinstance(event, ToolCallEvent):
                tool_calls_count += 1
            elif isinstance(event, AssistantChunkEvent):
                assistant_text += event.text or ""

            if conversation is not None:
                conversation.append(self._event_to_dict(event))

        return assistant_text, tool_calls_count

    def _normalize_event(self, event: AgentEvent) -> AgentEvent:
        """兼容 send_message_stream 可能返回字典的事件."""
        if isinstance(event, dict):
            event_type = event.get("type", "")
            if event_type == "tool_call":
                return ToolCallEvent(
                    name=event.get("name", ""), args=event.get("args", {})
                )
            if event_type == "assistant_chunk":
                return AssistantChunkEvent(text=event.get("text", ""))
        return event

    def _event_to_dict(self, event: AgentEvent) -> Dict[str, Any]:
        """把事件转换为可序列化的字典."""
        if isinstance(event, (AssistantChunkEvent, AssistantEndEvent)):
            return {"type": event.type, "text": getattr(event, "text", "")}
        if isinstance(event, ToolCallEvent):
            return {"type": event.type, "name": event.name, "args": event.args}
        if isinstance(event, UserInputEvent):
            return {"type": event.type, "text": event.text}
        return {"type": getattr(event, "type", "unknown")}

    def _run_evaluation(
        self, evaluation: EvaluationSpec, work_dir: Path
    ) -> tuple[int, str]:
        """执行判题命令，返回 (exit_code, output)."""
        result = subprocess.run(
            evaluation.command,
            cwd=str(work_dir),
            shell=True,
            capture_output=True,
            text=True,
            timeout=evaluation.timeout,
        )
        output = result.stdout + ("\n" + result.stderr if result.stderr else "")
        return result.returncode, output

    def _build_feedback(self, test_output: str) -> str:
        """测试未通过时，构建下一轮给 agent 的反馈."""
        return (
            "测试未通过，请根据下面的测试输出修复代码。"
            "只修改必要的内容，然后保存文件。\n\n"
            f"```\n{test_output}\n```"
        )

    def _truncate(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len] + "\n...[truncated]"

    def _print_task_result(self, result: TaskResult) -> None:
        mark = "[PASS]" if result.passed else "[FAIL]"
        self._safe_print(
            f"{mark} {result.name}: {result.status} | "
            f"{result.duration_seconds:.1f}s | "
            f"rounds={result.actual_rounds} | "
            f"tools={result.tool_calls_count} | "
            f"tokens={result.token_usage}"
        )

    @staticmethod
    def _safe_print(text: str) -> None:
        try:
            print(text)
        except UnicodeEncodeError:
            encoded = text.encode(sys.stdout.encoding or "utf-8", errors="replace")
            sys.stdout.buffer.write(encoded + b"\n")
