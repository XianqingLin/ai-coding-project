"""Git 版本控制工具.

提供结构化的 Git 操作能力, 包括状态查看、差异比较、历史记录、分支管理、
暂存、提交和推送. 所有写操作默认需要用户审批.
"""

from __future__ import annotations

import subprocess
from typing import Any, Dict, List, Optional

from ai_coding.tools.base import Tool, ToolParameter, ToolResult
from ai_coding.tools.safety import PathBoundaryError, resolve_workdir_cwd


def _ok(data: str, metadata: Optional[Dict[str, Any]] = None) -> ToolResult:
    return ToolResult.ok(data, metadata=metadata)


def _fail(data: str, error_code: Optional[str] = None) -> ToolResult:
    return ToolResult.fail(data, error_code=error_code)


def _wrap_git_result(result: str) -> ToolResult:
    """包装 _run_git 返回的字符串为 ToolResult."""
    if result.startswith("[错误]"):
        # 路径越界 / 工作目录解析失败归为 PATH_BOUNDARY_ERROR，其余归为 GIT_ERROR
        if "PathBoundaryError" in result or "超出工作目录" in result:
            return _fail(result, error_code="PATH_BOUNDARY_ERROR")
        return _fail(result, error_code="GIT_ERROR")
    return _ok(result)


MAX_OUTPUT_LINES = 200
MAX_OUTPUT_BYTES = 32 * 1024


def _truncate_output(text: str) -> str:
    """截断过长的输出, 避免超出 LLM 上下文窗口."""
    truncated = False
    lines = text.splitlines()
    if len(lines) > MAX_OUTPUT_LINES:
        lines = lines[:MAX_OUTPUT_LINES]
        truncated = True

    result = "\n".join(lines)
    if len(result.encode("utf-8")) > MAX_OUTPUT_BYTES:
        result = result.encode("utf-8")[:MAX_OUTPUT_BYTES].decode(
            "utf-8", errors="ignore"
        )
        truncated = True

    if truncated:
        result += "\n（已截断，请用更具体的参数查看）"
    return result


class GitToolBase(Tool):
    """Git 工具基类, 提供统一的 Git 命令执行能力."""

    requires_approval: bool = False

    def _resolve_cwd(self, cwd: Optional[str]) -> str:
        """解析并校验工作目录位于允许范围内."""
        if cwd is None:
            return str(resolve_workdir_cwd(None, self.work_dir))
        try:
            return str(resolve_workdir_cwd(cwd, self.work_dir))
        except PathBoundaryError as e:
            raise PathBoundaryError(str(e)) from e

    def _run_git(
        self,
        args: List[str],
        cwd: Optional[str] = None,
        check_repo: bool = True,
        input_text: Optional[str] = None,
    ) -> str:
        """执行 Git 命令并返回格式化结果."""
        try:
            effective_cwd = self._resolve_cwd(cwd)
        except PathBoundaryError as e:
            return f"[错误] {e}"
        except Exception as e:
            return f"[错误] 解析工作目录失败: {e}"

        if check_repo:
            repo_check = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=str(effective_cwd),
                capture_output=True,
                text=True,
            )
            if repo_check.returncode != 0:
                return "[错误] 当前目录不是 git 仓库"

        cmd = ["git", "--no-pager", *args]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(effective_cwd),
                capture_output=True,
                text=True,
                input=input_text,
            )
        except FileNotFoundError:
            return "[错误] 未找到 git 命令, 请确保 Git 已安装并加入 PATH"
        except Exception as e:
            return f"[错误] 执行 git 命令失败: {e}"

        if proc.returncode != 0:
            stderr = proc.stderr.strip() if proc.stderr else ""
            msg = f"[错误] git {' '.join(args)} 失败（退出码 {proc.returncode}）"
            if stderr:
                msg += f": {stderr}"
            return msg

        output = proc.stdout
        if not output:
            return "[成功] 命令执行完成, 无输出."
        return _truncate_output(output)


class GitStatusTool(GitToolBase):
    """查看 Git 仓库状态."""

    name = "git_status"
    description = (
        "查看 git 仓库当前工作区与暂存区的状态, "
        "返回变更文件的列表及状态码（M 修改、A 新增、D 删除、?? 未跟踪等）."
    )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "cwd",
                "string",
                "要查看的目录路径, 默认当前工作目录",
                required=False,
            ),
        ]

    def execute(  # type: ignore[override]
        self, cwd: Optional[str] = None
    ) -> ToolResult:
        return _wrap_git_result(
            self._run_git(["status", "--porcelain=v1", "-uall"], cwd=cwd)
        )


class GitDiffTool(GitToolBase):
    """查看 Git 差异."""

    name = "git_diff"
    description = (
        "查看 git 工作区或暂存区与 HEAD 之间的差异. "
        "用于在修改代码后了解具体变更内容."
    )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "cached",
                "boolean",
                "是否查看已暂存（staged）的变更, 默认 false",
                required=False,
                default=False,
            ),
            ToolParameter(
                "path",
                "string",
                "限制差异范围的文件或目录路径",
                required=False,
            ),
            ToolParameter(
                "cwd",
                "string",
                "工作目录, 默认当前工作目录",
                required=False,
            ),
        ]

    def execute(  # type: ignore[override]
        self,
        cached: bool = False,
        path: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> ToolResult:
        args = ["diff"]
        if cached:
            args.append("--cached")
        if path:
            args.extend(["--", path])
        return _wrap_git_result(self._run_git(args, cwd=cwd))


class GitLogTool(GitToolBase):
    """查看 Git 提交历史."""

    name = "git_log"
    description = "查看 git 提交历史, 支持限制条数、时间范围和路径过滤."

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "limit",
                "integer",
                "返回的最大提交条数, 默认 20",
                required=False,
                default=20,
            ),
            ToolParameter(
                "since",
                "string",
                "仅显示指定时间之后的提交, 例如 '1 week ago'",
                required=False,
            ),
            ToolParameter(
                "path",
                "string",
                "限制日志范围的文件或目录路径",
                required=False,
            ),
            ToolParameter(
                "cwd",
                "string",
                "工作目录, 默认当前工作目录",
                required=False,
            ),
        ]

    def execute(  # type: ignore[override]
        self,
        limit: int = 20,
        since: Optional[str] = None,
        path: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> ToolResult:
        args = ["log", "--oneline", f"-n {max(1, int(limit))}"]
        if since:
            args.extend(["--since", since])
        if path:
            args.extend(["--", path])
        return _wrap_git_result(self._run_git(args, cwd=cwd))


class GitBranchListTool(GitToolBase):
    """列出 Git 分支."""

    name = "git_branch_list"
    description = "列出当前 git 仓库的所有本地分支, 当前分支会带 * 标记."

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "cwd",
                "string",
                "工作目录, 默认当前工作目录",
                required=False,
            ),
        ]

    def execute(  # type: ignore[override]
        self, cwd: Optional[str] = None
    ) -> ToolResult:
        return _wrap_git_result(self._run_git(["branch", "--list"], cwd=cwd))


class GitBranchCreateTool(GitToolBase):
    """创建 Git 分支."""

    name = "git_branch_create"
    requires_approval = True
    description = (
        "在 git 仓库中创建新分支, 创建后不会自动切换. "
        "此工具会修改仓库状态, 需要用户审批."
    )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("branch", "string", "要创建的分支名称"),
            ToolParameter(
                "base",
                "string",
                "基于哪个分支或提交创建, 默认当前 HEAD",
                required=False,
            ),
            ToolParameter(
                "cwd",
                "string",
                "工作目录, 默认当前工作目录",
                required=False,
            ),
        ]

    def execute(  # type: ignore[override]
        self,
        branch: str,
        base: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> ToolResult:
        if not branch.strip():
            return _fail("[错误] branch 参数不能为空", error_code="VALIDATION_ERROR")
        args = ["branch", branch]
        if base:
            args.append(base)
        return _wrap_git_result(self._run_git(args, cwd=cwd))


class GitBranchSwitchTool(GitToolBase):
    """切换 Git 分支."""

    name = "git_branch_switch"
    requires_approval = True
    description = (
        "切换到指定的 git 分支. 如果分支不存在且 create=true, "
        "则会创建并切换. 此工具会修改工作区文件, 需要用户审批."
    )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("branch", "string", "要切换到的分支名称"),
            ToolParameter(
                "create",
                "boolean",
                "分支不存在时是否创建, 默认 false",
                required=False,
                default=False,
            ),
            ToolParameter(
                "cwd",
                "string",
                "工作目录, 默认当前工作目录",
                required=False,
            ),
        ]

    def execute(  # type: ignore[override]
        self,
        branch: str,
        create: bool = False,
        cwd: Optional[str] = None,
    ) -> ToolResult:
        if not branch.strip():
            return _fail("[错误] branch 参数不能为空", error_code="VALIDATION_ERROR")
        args = ["switch"]
        if create:
            args.append("-c")
        args.append(branch)
        return _wrap_git_result(self._run_git(args, cwd=cwd))


class GitAddTool(GitToolBase):
    """将文件变更暂存到 Git 索引."""

    name = "git_add"
    requires_approval = True
    description = (
        "将指定文件或目录的变更添加到 git 暂存区, 为提交做准备. "
        "此工具会修改仓库状态, 需要用户审批."
    )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "paths",
                "string",
                "要暂存的文件或目录路径, 多个路径用逗号或空格分隔",
            ),
            ToolParameter(
                "cwd",
                "string",
                "工作目录, 默认当前工作目录",
                required=False,
            ),
        ]

    def execute(  # type: ignore[override]
        self, paths: str, cwd: Optional[str] = None
    ) -> ToolResult:
        if not paths.strip():
            return _fail("[错误] paths 参数不能为空", error_code="VALIDATION_ERROR")
        raw_paths = [p.strip() for p in paths.replace(",", " ").split() if p.strip()]
        return _wrap_git_result(self._run_git(["add", *raw_paths], cwd=cwd))


class GitCommitTool(GitToolBase):
    """创建 Git 提交."""

    name = "git_commit"
    requires_approval = True
    description = (
        "使用给定提交信息创建 git 提交. "
        "请确保已先用 git_add 暂存需要提交的变更. "
        "此工具会修改仓库历史, 需要用户审批."
    )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("message", "string", "提交信息"),
            ToolParameter(
                "cwd",
                "string",
                "工作目录, 默认当前工作目录",
                required=False,
            ),
        ]

    def execute(  # type: ignore[override]
        self, message: str, cwd: Optional[str] = None
    ) -> ToolResult:
        if not message.strip():
            return _fail("[错误] 提交信息不能为空", error_code="VALIDATION_ERROR")
        return _wrap_git_result(
            self._run_git(
                ["commit", "-m", message],
                cwd=cwd,
                input_text="\n",
            )
        )


class GitPushTool(GitToolBase):
    """推送 Git 提交到远程仓库."""

    name = "git_push"
    requires_approval = True
    description = (
        "将本地 git 提交推送到远程仓库. "
        "此工具涉及网络操作和远程仓库写入, 需要用户审批."
    )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "remote",
                "string",
                "远程仓库名称, 默认 origin",
                required=False,
                default="origin",
            ),
            ToolParameter(
                "branch",
                "string",
                "要推送的本地分支名, 默认当前分支",
                required=False,
            ),
            ToolParameter(
                "cwd",
                "string",
                "工作目录, 默认当前工作目录",
                required=False,
            ),
        ]

    def execute(  # type: ignore[override]
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> ToolResult:
        remote = remote or "origin"
        args = ["push", remote]
        if branch:
            args.append(branch)
        return _wrap_git_result(self._run_git(args, cwd=cwd))
