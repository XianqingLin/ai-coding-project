"""Shell 命令执行相关工具.

提供在系统 shell 中执行命令的能力，支持前台同步执行和后台异步执行.
"""

import atexit
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from ai_coding.tools.base import Tool, ToolParameter
from ai_coding.tools.sandbox import SandboxViolationError, resolve_sandboxed_cwd


class ExecuteCommandTool(Tool):
    """执行 shell 命令，支持前台同步和后台异步两种模式."""

    name = "execute_command"
    requires_approval = True
    description = (
        "执行 shell 命令. 支持前台同步执行和后台异步执行两种模式.\n"
        "前台模式：等待命令完成并返回输出，适合运行测试、安装依赖等.\n"
        "后台模式：立即返回任务 ID，命令在后台持续运行，适合启动服务器等长驻进程.\n"
        "后台任务可通过 task_list / task_output / task_stop 工具管理."
    )

    # 保留最近 50 个已终止任务的日志文件
    MAX_LOG_RETENTION = 50
    # 前台超时上限 5 分钟 = 300000 毫秒
    FOREGROUND_MAX_TIMEOUT_MS = 300000

    def __init__(self) -> None:
        super().__init__()
        self._bg_tasks: Dict[str, dict] = {}
        atexit.register(self._cleanup_all_on_exit)

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("command", "string", "要执行的 shell 命令"),
            ToolParameter("cwd", "string", "工作目录，默认当前目录", required=False),
            ToolParameter(
                "timeout",
                "integer",
                "超时时间（毫秒），前台默认 60000，后台默认 60000",
                required=False,
                default=60000,
            ),
            ToolParameter(
                "run_in_background",
                "boolean",
                "是否后台异步执行，默认 false",
                required=False,
                default=False,
            ),
            ToolParameter(
                "description",
                "string",
                "后台任务描述，run_in_background=true 时必填",
                required=False,
            ),
            ToolParameter(
                "disable_timeout",
                "boolean",
                "后台任务是否取消超时限制，默认 false",
                required=False,
                default=False,
            ),
        ]

    def execute(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 60000,
        run_in_background: bool = False,
        description: Optional[str] = None,
        disable_timeout: bool = False,
    ) -> str:
        if run_in_background:
            return self._run_background(command, cwd, description, disable_timeout)
        return self._run_foreground(command, cwd, timeout)

    def _run_foreground(self, command: str, cwd: Optional[str], timeout_ms: int) -> str:
        """前台同步执行：两阶段终止策略 (SIGTERM -> 5s -> SIGKILL)."""
        timeout_ms = min(timeout_ms, self.FOREGROUND_MAX_TIMEOUT_MS)
        timeout_sec = timeout_ms / 1000.0 if timeout_ms > 0 else None

        try:
            effective_cwd = resolve_sandboxed_cwd(cwd, self.work_dir)
        except SandboxViolationError as e:
            return f"[错误] {e}"
        except Exception as e:
            return f"[错误] 解析工作目录失败: {e}"

        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=str(effective_cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            return f"[错误] 启动命令失败: {e}"

        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            # 两阶段终止
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            return f"[超时] 命令执行超时（{timeout_ms}ms），已强制终止."

        output_parts = []
        if stdout:
            output_parts.append(f"[stdout]\n{stdout}")
        if stderr:
            output_parts.append(f"[stderr]\n{stderr}")
        if proc.returncode != 0:
            output_parts.append(f"[退出码] {proc.returncode}")

        if not output_parts:
            return "[成功] 命令执行完成, 无输出."
        return "\n\n".join(output_parts)

    def _run_background(
        self,
        command: str,
        cwd: Optional[str],
        description: Optional[str],
        disable_timeout: bool,
    ) -> str:
        """后台异步执行：立即返回任务 ID."""
        if not description:
            return "[错误] run_in_background=true 时必须提供 description 参数."

        try:
            effective_cwd = resolve_sandboxed_cwd(cwd, self.work_dir)
        except SandboxViolationError as e:
            return f"[错误] {e}"
        except Exception as e:
            return f"[错误] 解析工作目录失败: {e}"

        task_id = uuid.uuid4().hex[:8]
        output_path = Path(tempfile.gettempdir()) / f"ai-coding-bg-{task_id}.log"

        try:
            f = open(output_path, "w", encoding="utf-8")
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=str(effective_cwd),
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as e:
            return f"[错误] 启动后台任务失败: {e}"

        self._bg_tasks[task_id] = {
            "task_id": task_id,
            "command": command,
            "description": description,
            "status": "running",
            "pid": proc.pid,
            "output_path": str(output_path),
            "start_time": time.time(),
            "end_time": None,
            "exit_code": None,
            "_proc": proc,
            "_file": f,
        }

        return (
            f"[后台任务已启动] {task_id}\n"
            f"命令: {command}\n"
            f"描述: {description}\n"
            f"输出: {output_path}"
        )

    # ------------------------------------------------------------------
    # 以下方法供 TaskListTool / TaskOutputTool / TaskStopTool 调用
    # ------------------------------------------------------------------

    def list_tasks(self, active_only: bool = True, limit: int = 20) -> List[dict]:
        """返回后台任务列表（可序列化的 dict 列表，不含 _proc/_file）."""
        self._refresh_running_status()
        tasks = list(self._bg_tasks.values())
        if active_only:
            tasks = [t for t in tasks if t["status"] == "running"]
        # 按 start_time 倒序
        tasks.sort(key=lambda t: t["start_time"], reverse=True)
        return [self._serialize(t) for t in tasks[:limit]]

    def get_task(self, task_id: str) -> Optional[dict]:
        """获取单个任务（含懒刷新状态）."""
        self._refresh_running_status()
        t = self._bg_tasks.get(task_id)
        return self._serialize(t) if t else None

    def read_task_output(self, task_id: str, max_bytes: int = 32 * 1024) -> str:
        """读取任务输出文件末尾 max_bytes 内容."""
        t = self._bg_tasks.get(task_id)
        if not t:
            return f"[错误] 任务不存在: {task_id}"
        path = Path(t["output_path"])
        if not path.exists():
            return "[信息] 输出文件尚未生成或已被清理."
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size > max_bytes:
                    f.seek(-max_bytes, os.SEEK_END)
                    data = f.read()
                    # 去掉可能截断的首个字节（确保从完整字符开始）
                    try:
                        return data.decode("utf-8")
                    except UnicodeDecodeError:
                        return data[1:].decode("utf-8")
                else:
                    f.seek(0)
                    return f.read().decode("utf-8")
        except Exception as e:
            return f"[错误] 读取输出失败: {e}"

    def stop_task(self, task_id: str, reason: str = "Stopped by TaskStop") -> str:
        """停止指定后台任务，两阶段终止."""
        t = self._bg_tasks.get(task_id)
        if not t:
            return f"[错误] 任务不存在: {task_id}"

        if t["status"] != "running":
            return f"[信息] 任务 {task_id} 已处于终止状态 ({t['status']})，无需操作."

        proc = t.get("_proc")
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

            if "_file" in t:
                t["_file"].close()
                del t["_file"]
            del t["_proc"]

        t["status"] = "stopped"
        t["end_time"] = time.time()
        t["exit_code"] = -1
        return f"[停止] {task_id}: {reason}"

    def background_tasks(self) -> List[dict]:
        """返回所有任务的可序列化版本（供 tools_node 同步到 AgentState）."""
        self._refresh_running_status()
        self._cleanup_old_logs()
        return [self._serialize(t) for t in self._bg_tasks.values()]

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _refresh_running_status(self) -> None:
        """扫描所有 running 任务，更新已终止任务的状态."""
        for t in list(self._bg_tasks.values()):
            if t["status"] == "running" and "_proc" in t:
                proc = t["_proc"]
                ret = proc.poll()
                if ret is not None:
                    t["status"] = "completed" if ret == 0 else "failed"
                    t["end_time"] = time.time()
                    t["exit_code"] = ret
                    if "_file" in t:
                        t["_file"].close()
                        del t["_file"]
                    del t["_proc"]

    def _serialize(self, t: dict) -> dict:
        """去掉内部 _proc / _file 引用，返回可序列化的 dict."""
        return {k: v for k, v in t.items() if not k.startswith("_")}

    def _cleanup_old_logs(self) -> None:
        """清理超出保留数量的已终止任务日志."""
        terminated = [
            (tid, t) for tid, t in self._bg_tasks.items() if t["status"] != "running"
        ]
        if len(terminated) <= self.MAX_LOG_RETENTION:
            return
        # 按 end_time 升序，保留最新的 MAX_LOG_RETENTION 个
        terminated.sort(key=lambda x: x[1].get("end_time") or 0)
        to_remove = terminated[: -self.MAX_LOG_RETENTION]
        for tid, t in to_remove:
            try:
                Path(t["output_path"]).unlink(missing_ok=True)
            except Exception:
                pass
            del self._bg_tasks[tid]

    def _cleanup_all_on_exit(self) -> None:
        """Agent 进程退出时自动清理所有 running 任务."""
        for t in list(self._bg_tasks.values()):
            if t.get("status") == "running" and "_proc" in t:
                proc = t["_proc"]
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                except Exception:
                    pass
                if "_file" in t:
                    try:
                        t["_file"].close()
                    except Exception:
                        pass
                t["status"] = "stopped"
                t["end_time"] = time.time()
