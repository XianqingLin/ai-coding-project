"""Grep 搜索工具.

在代码库中按正则表达式搜索文本, 快速定位需要修改的代码位置.
"""

import fnmatch
import os
import re
from typing import List

from ai_coding.tools.base import Tool, ToolParameter


class GrepTool(Tool):
    """在代码库中搜索文本模式."""

    name = "grep"
    description = (
        "在代码库中按正则表达式搜索文本. 用于快速定位函数、变量、字符串等出现的位置.\n"
        "自动排除 .git/, node_modules/, vendor/ 等目录.\n"
        "最多返回 50 条匹配; 结果过多时请缩小 pattern 或加 glob 过滤."
    )

    # 默认排除的目录
    DEFAULT_EXCLUDE_DIRS = {
        ".git", ".github", "node_modules", "vendor",
        "__pycache__", ".venv", "venv", ".tox",
        "build", "dist", ".pytest_cache", ".mypy_cache",
        "target",  # Rust
    }

    # 默认排除的文件模式
    DEFAULT_EXCLUDE_GLOBS = {
        "*.min.js", "*.min.css", "*.map",
        "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico",
        "*.pdf", "*.zip", "*.tar", "*.gz", "*.rar",
        "*.exe", "*.dll", "*.so", "*.dylib",
        "*.wasm", "*.woff", "*.woff2", "*.ttf",
    }

    MAX_RESULTS = 50
    MAX_LINE_LEN = 500

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("pattern", "string", "搜索模式(Python 正则表达式)"),
            ToolParameter("path", "string", "搜索路径, 默认当前目录", required=False, default="."),
            ToolParameter("glob", "string", "文件过滤器, 如 '*.go' 或 '*.{py,js}'", required=False),
            ToolParameter("output_mode", "string", "content(显示匹配行) 或 files(只显示文件列表)", required=False, default="content"),
        ]

    def execute(self, pattern: str, path: str = ".", glob: str = None, output_mode: str = "content") -> str:
        if not os.path.exists(path):
            return f"[错误] 路径不存在: {path}"

        if not os.path.isdir(path):
            # 如果是文件, 直接搜索该文件
            return self._search_file(path, pattern, output_mode)

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"[错误] 正则表达式无效: {e}"

        # 解析 glob 过滤
        globs = self._parse_glob(glob)
        results = []
        files_matched = set()

        for root, dirs, files in os.walk(path):
            # 排除目录
            dirs[:] = [
                d for d in dirs
                if d not in self.DEFAULT_EXCLUDE_DIRS
                and not d.startswith(".")
            ]

            for filename in files:
                # 排除文件
                if any(fnmatch.fnmatch(filename, g) for g in self.DEFAULT_EXCLUDE_GLOBS):
                    continue
                if filename.startswith("."):
                    continue
                # glob 过滤
                if globs and not any(fnmatch.fnmatch(filename, g) for g in globs):
                    continue

                filepath = os.path.join(root, filename)
                file_results = self._search_file_lines(filepath, regex, output_mode)

                if output_mode == "files":
                    if file_results:
                        files_matched.add(filepath)
                        if len(files_matched) >= self.MAX_RESULTS:
                            break
                else:
                    results.extend(file_results)
                    if len(results) >= self.MAX_RESULTS:
                        break

            if (output_mode == "files" and len(files_matched) >= self.MAX_RESULTS) or \
               (output_mode != "files" and len(results) >= self.MAX_RESULTS):
                break

        if output_mode == "files":
            if not files_matched:
                return f"未找到匹配文件 (pattern={pattern!r})"
            lines = sorted(f.replace(os.sep, "/") for f in files_matched)
            if len(lines) >= self.MAX_RESULTS:
                lines.append(f"... 结果超过 {self.MAX_RESULTS} 条, 已截断。请缩小搜索范围。")
            return "匹配文件:\n" + "\n".join(lines)

        # content mode
        if not results:
            return f"未找到匹配 (pattern={pattern!r})"

        lines = []
        for filepath, line_no, line_text in results[:self.MAX_RESULTS]:
            rel_path = os.path.relpath(filepath, path).replace(os.sep, "/")
            # 截断超长行
            if len(line_text) > self.MAX_LINE_LEN:
                line_text = line_text[:self.MAX_LINE_LEN] + " ..."
            lines.append(f"{rel_path}:{line_no} | {line_text}")

        if len(results) >= self.MAX_RESULTS:
            lines.append(f"... 结果超过 {self.MAX_RESULTS} 条, 已截断。请缩小 pattern 或加 glob 过滤。")

        return "\n".join(lines)

    def _parse_glob(self, glob_str: str = None) -> List[str]:
        """解析 glob 字符串. 支持 '{a,b,c}' 语法."""
        if not glob_str:
            return []
        glob_str = glob_str.strip()
        if glob_str.startswith("{") and glob_str.endswith("}"):
            # {*.go,*.js}
            return [g.strip() for g in glob_str[1:-1].split(",")]
        return [glob_str]

    def _search_file(self, filepath: str, pattern: str, output_mode: str) -> str:
        """搜索单个文件(外部接口, path 为文件时调用)."""
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"[错误] 正则表达式无效: {e}"
        results = self._search_file_lines(filepath, regex, output_mode)
        if output_mode == "files":
            return filepath if results else ""
        if not results:
            return f"未找到匹配 (pattern={pattern!r})"
        lines = []
        for _, line_no, line_text in results:
            if len(line_text) > self.MAX_LINE_LEN:
                line_text = line_text[:self.MAX_LINE_LEN] + " ..."
            lines.append(f"{filepath.replace(os.sep, '/' )}:{line_no} | {line_text}")
        return "\n".join(lines)

    def _search_file_lines(self, filepath: str, regex: re.Pattern, output_mode: str) -> List:
        """搜索文件内容, 返回 [(filepath, line_no, line_text), ...]."""
        results = []
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f, start=1):
                    if regex.search(line):
                        results.append((filepath, line_no, line.rstrip("\n")))
                        if output_mode != "files" and len(results) >= self.MAX_RESULTS:
                            break
        except (UnicodeDecodeError, OSError):
            # 二进制文件或无法读取, 跳过
            pass
        return results
