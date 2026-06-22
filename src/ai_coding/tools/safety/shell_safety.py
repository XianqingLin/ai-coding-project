"""Shell 命令安全校验.

在路径边界校验之外，再对命令字符串做静态语义分析，拦截常见高危模式
（强制删除根目录、权限提升、远程管道执行、向系统目录写入、cd .. 等）。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


class ShellSafetyError(ValueError):
    """命令违反安全策略."""

    pass


@dataclass
class ShellSafetyChecker:
    """对 shell 命令做静态安全检查.

    默认策略包括：
    - 破坏性操作：rm -rf /、mkfs、fdisk、dd 原始设备等
    - 权限提升：sudo、su -、doas
    - 远程代码执行：curl/wget/fetch 直接管道到 shell
    - 向系统目录/块设备重定向
    - cd .. 跳出工作目录
    - 破坏性命令引用工作目录外的敏感绝对路径或包含 .. 遍历
    """

    work_dir: str = ""
    blocked_patterns: Optional[List[Tuple[re.Pattern, str]]] = None
    blocked_commands: Optional[set] = None
    sensitive_abs_paths: Optional[set] = None

    def __post_init__(self) -> None:
        if self.blocked_patterns is None:
            self.blocked_patterns = self._default_patterns()
        if self.blocked_commands is None:
            self.blocked_commands = self._default_blocked_commands()
        if self.sensitive_abs_paths is None:
            self.sensitive_abs_paths = self._default_sensitive_paths()

    @staticmethod
    def _default_patterns() -> List[Tuple[re.Pattern, str]]:
        return [
            # 1. 破坏性文件操作
            (
                re.compile(r"\brm\s+-[a-zA-Z]*f\s+(/?\s*$|/\s+|\*/|/\*)", re.I),
                "禁止强制递归删除根目录或系统通配路径",
            ),
            (re.compile(r"\bmkfs\b", re.I), "禁止格式化文件系统"),
            (re.compile(r"\bfdisk\b", re.I), "禁止磁盘分区操作"),
            (
                re.compile(r"\bdd\s+if\s*=\s*/dev/(sd|hd|nvme|mmcblk)"),
                "禁止对原始磁盘设备执行 dd",
            ),
            (
                re.compile(r"\bchmod\s+-R\s+777\s*/", re.I),
                "禁止递归修改系统目录权限",
            ),
            (
                re.compile(r"\bchown\s+-R\s+\/", re.I),
                "禁止递归修改系统目录属主",
            ),
            # 2. 权限提升
            (re.compile(r"\bsudo\b", re.I), "禁止使用 sudo 提权"),
            (re.compile(r"\bsu\s+-", re.I), "禁止使用 su 切换用户"),
            (re.compile(r"\bdoas\b", re.I), "禁止使用 doas 提权"),
            # 3. 远程代码执行：curl/wget/fetch 直接管道到 shell
            (
                re.compile(r"\b(curl|wget|fetch)\s+[^\|]*\|\s*(ba)?sh\b", re.I),
                "禁止通过管道直接执行远程脚本",
            ),
            # 4. 向块设备/系统目录重定向
            (
                re.compile(r"[<>]\s*/dev/(sd|hd|nvme|mmcblk)"),
                "禁止重定向到块设备",
            ),
            (
                re.compile(r"[<>]\s*/(etc|sys|proc|dev|boot|root)(/|$)"),
                "禁止直接写入系统目录",
            ),
            # 5. cd .. 跳出工作目录
            (
                re.compile(r"\bcd\s+\.\.(\s|$|;|\||&)"),
                "禁止通过 cd .. 跳出工作目录",
            ),
        ]

    @staticmethod
    def _default_blocked_commands() -> set:
        return {"shutdown", "reboot", "halt", "poweroff", "init", "systemctl"}

    @staticmethod
    def _default_sensitive_paths() -> set:
        return {
            "/etc",
            "/usr/etc",
            "/sys",
            "/proc",
            "/dev",
            "/boot",
            "/root",
            "/home",
            "/var",
            "C:\\Windows",
        }

    # ---------- 路径检测辅助 ----------

    @staticmethod
    def _command_tokens(command: str) -> List[str]:
        """按空白和常见 shell 分隔符拆分命令，忽略空 token."""
        return [
            t.strip("\"'")
            for t in re.split(r"[\s;|\&<>()]+", command)
            if t.strip("\"'")
        ]

    @staticmethod
    def _extract_path_tokens(command: str) -> List[str]:
        """从命令中提取可能是绝对路径的 token，忽略 - 开关."""
        paths: List[str] = []
        for token in ShellSafetyChecker._command_tokens(command):
            if token.startswith("-"):
                continue
            if token.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", token):
                paths.append(token)
        return paths

    def _work_dir_path(self) -> Optional[Path]:
        wd = self.work_dir or os.getcwd()
        try:
            return Path(wd).expanduser().resolve()
        except Exception:
            return None

    @staticmethod
    def _is_destructive_command(command: str) -> bool:
        """判断命令是否具有破坏性/写入性."""
        destructive = {
            "rm",
            "rmdir",
            "chmod",
            "chown",
            "mv",
            "cp",
            "dd",
            "mkfs",
            "mkfs.ext4",
            "mkfs.ntfs",
            "mount",
            "umount",
            "truncate",
            "shred",
        }
        first = command.split(None, 1)[0].lower()
        # 处理类似 /bin/rm 的情况
        if first.startswith(("/bin/", "/usr/bin/", "/usr/sbin/", "/sbin/")):
            first = first.split("/")[-1]
        return first in destructive or ">" in command

    def _check_path_traversal(self, command: str) -> Optional[str]:
        """检测破坏性命令中是否包含 .. 路径遍历."""
        if not self._is_destructive_command(command):
            return None

        for token in self._command_tokens(command):
            if token.startswith("-"):
                continue
            if ".." in token:
                return f"破坏性命令包含路径遍历: {token}"
        return None

    def _check_sensitive_paths(self, command: str) -> Optional[str]:
        """检测破坏性命令是否引用了工作目录外的敏感系统路径."""
        if not self._is_destructive_command(command):
            return None

        work = self._work_dir_path()
        work_str = str(work).lower() if work else ""

        for token in self._extract_path_tokens(command):
            # Unix 风格绝对路径：直接用字符串前缀匹配，避免 Windows Path 解析差异
            if token.startswith("/"):
                lower_token = token.lower()
                if work_str and lower_token.startswith(work_str):
                    continue
                for prefix in self.sensitive_abs_paths or set():
                    if prefix.startswith("/") and lower_token.startswith(
                        prefix.lower()
                    ):
                        return f"命令引用了工作目录外的敏感路径: {token}"
                continue

            # Windows 风格绝对路径：用 Path 解析
            try:
                p = Path(token).expanduser().resolve()
            except Exception:
                continue

            if work and (p == work or p.is_relative_to(work)):
                continue

            # 放行已知的可执行目录
            try:
                parts = p.parts
                if parts[:3] in [
                    ("/", "usr", "bin"),
                    ("/", "usr", "sbin"),
                    ("/", "usr", "local", "bin"),
                ] or parts[:2] in [("/", "bin"), ("/", "sbin")]:
                    continue
            except Exception:
                pass

            lower = str(p).lower()
            for prefix in self.sensitive_abs_paths or set():
                if lower.startswith(prefix.lower()):
                    return f"命令引用了工作目录外的敏感路径: {token}"
        return None

    # ---------- 主入口 ----------

    def check(self, command: str, cwd: Optional[str] = None) -> None:
        """校验命令，违反策略时抛出 ShellSafetyError."""
        if cwd:
            self.work_dir = cwd

        cmd = command.strip()
        if not cmd:
            raise ShellSafetyError("命令不能为空")

        # 危险正则
        for pattern, reason in self.blocked_patterns or []:
            if pattern.search(cmd):
                raise ShellSafetyError(f"安全策略命中: {reason}")

        # 危险命令名
        first = cmd.split(None, 1)[0]
        if first.lower() in (self.blocked_commands or set()):
            raise ShellSafetyError(f"禁止执行的系统命令: {first}")

        # 路径遍历
        traversal_reason = self._check_path_traversal(cmd)
        if traversal_reason:
            raise ShellSafetyError(traversal_reason)

        # 敏感绝对路径
        sensitive_reason = self._check_sensitive_paths(cmd)
        if sensitive_reason:
            raise ShellSafetyError(sensitive_reason)

    def is_safe(self, command: str, cwd: Optional[str] = None) -> Tuple[bool, str]:
        """返回 (是否安全, 原因)."""
        try:
            self.check(command, cwd=cwd)
            return True, ""
        except ShellSafetyError as e:
            return False, str(e)
