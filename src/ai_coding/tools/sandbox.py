"""路径沙箱工具.

限制工具只能访问当前工作目录内的文件和目录，防止 LLM 通过绝对路径
或 .. 遍历访问工作区之外的文件系统。
"""

import os
from pathlib import Path
from typing import Optional


class SandboxViolationError(ValueError):
    """路径超出工作目录沙箱范围时抛出."""

    pass


def _is_path_within_work_dir(target: Path, work: Path) -> bool:
    """判断 target 是否位于 work 目录内（包含 work 本身）."""
    try:
        target.relative_to(work)
        return True
    except ValueError:
        return False


def resolve_sandboxed_path(
    path: str,
    work_dir: str,
    must_exist: bool = False,
    allow_abs_within_work_dir: bool = True,
) -> Path:
    """解析路径并校验其是否位于工作目录沙箱内.

    Args:
        path: 用户传入的路径（相对或绝对）.
        work_dir: 工作目录根路径.
        must_exist: 是否要求路径已存在.
        allow_abs_within_work_dir: 是否允许绝对路径，只要解析后仍在 work_dir 内.

    Returns:
        解析后的绝对路径.

    Raises:
        SandboxViolationError: 路径超出工作目录范围.
        ValueError: 路径为空或无法解析.
    """
    if not path:
        raise ValueError("路径不能为空")

    effective_work_dir = work_dir or os.getcwd()
    work = Path(effective_work_dir).expanduser().resolve()
    if not work.is_dir():
        raise ValueError(f"工作目录无效: {effective_work_dir}")

    raw = os.path.expanduser(path)

    # 拒绝包含 .. 或 . 的相对遍历（先按字符串判断）
    parts = Path(raw).parts
    if ".." in parts:
        raise SandboxViolationError(f"路径包含 '..' 遍历，被拒绝: {path}")

    if os.path.isabs(raw):
        if not allow_abs_within_work_dir:
            raise SandboxViolationError(f"不允许使用绝对路径: {path}")
        target = Path(raw).resolve()
    else:
        target = (work / raw).resolve()

    if not _is_path_within_work_dir(target, work):
        raise SandboxViolationError(
            f"路径 '{path}' 超出工作目录 '{work_dir}' 范围，拒绝访问"
        )

    if must_exist and not target.exists():
        raise ValueError(f"路径不存在: {path}")

    return target


def resolve_sandboxed_cwd(
    cwd: Optional[str],
    work_dir: str,
) -> Path:
    """解析 shell 命令的 cwd 参数，限制在工作目录内.

    若 cwd 为空，则默认返回工作目录。
    """
    if not cwd:
        return Path(work_dir).expanduser().resolve()
    return resolve_sandboxed_path(cwd, work_dir, must_exist=True)
