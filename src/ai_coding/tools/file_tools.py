"""文件系统相关工具.

提供文件读取、写入、编辑、搜索、glob 匹配等基础操作.
"""

import difflib
import fnmatch
import glob
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ai_coding.tools.base import Tool, ToolParameter, ToolResult


def _ok(data: str, metadata: Optional[Dict[str, Any]] = None) -> ToolResult:
    return ToolResult.ok(data, metadata=metadata)


def _fail(data: str, error_code: Optional[str] = None) -> ToolResult:
    return ToolResult.fail(data, error_code=error_code)


class ReadFileTool(Tool):
    """读取文件内容."""

    name = "read_file"
    description = (
        "读取指定文件的内容, 返回文件中的文本. 用于查看代码、配置文件或文档内容.\n"
        "单次调用最多读取 300 行; 超过 300 行的文件需要分多次读取.\n"
        "超长行(>2000字符)会被截断. 配合 line_offset 和 n_lines 分段读取完整内容."
    )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("path", "string", "要读取的文件路径（相对路径或绝对路径）"),
            ToolParameter(
                "line_offset",
                "integer",
                "起始行号(从1开始), 默认1",
                required=False,
                default=1,
            ),
            ToolParameter(
                "n_lines",
                "integer",
                "读取行数, 默认300, 最大300",
                required=False,
                default=300,
            ),
        ]

    def execute(  # type: ignore[override]
        self, path: str, line_offset: int = 1, n_lines: int = 300
    ) -> ToolResult:
        ok, result = _resolve_path_or_error(self, path, must_exist=True)
        if not ok:
            return _fail(str(result), error_code="PATH_BOUNDARY_ERROR")
        path = str(result)

        if not os.path.exists(path):
            return _fail(f"[错误] 文件不存在: {path}", error_code="FILE_NOT_FOUND")

        if os.path.isdir(path):
            return _fail(
                f"[错误] '{path}' 是一个目录, 请使用 list_dir 工具查看目录内容.",
                error_code="FILE_NOT_FOUND",
            )

        try:
            line_offset = max(1, int(line_offset))
            n_lines = max(1, min(300, int(n_lines)))

            with open(path, "r", encoding="utf-8") as f:
                all_lines = f.read().split("\n")
            total_lines = len(all_lines)

            start_idx = line_offset - 1
            end_idx = start_idx + n_lines
            chunk = all_lines[start_idx:end_idx]

            # 超长行截断
            MAX_LINE_LEN = 2000
            truncated_lines = []
            for line in chunk:
                if len(line) > MAX_LINE_LEN:
                    line = line[:MAX_LINE_LEN] + " ..."
                truncated_lines.append(line)

            numbered = "\n".join(
                f"{start_idx + i + 1:4d} | {line}"
                for i, line in enumerate(truncated_lines)
            )

            result = f"文件: {path}\n{'='*50}\n{numbered}\n{'='*50}\n"
            segment_range = f"{start_idx + 1}-{min(end_idx, total_lines)}"
            result += f"(本段 {segment_range} / 共 {total_lines} 行"
            if total_lines > 1000:
                next_offset = min(end_idx + 1, total_lines)
                result += f", 超过 1000 行, 可用 line_offset={next_offset} 继续读取"
            result += ")"
            return _ok(result)

        except UnicodeDecodeError:
            return _fail(
                f"[错误] 无法以文本格式读取 '{path}', 可能是二进制文件.",
                error_code="EXECUTION_ERROR",
            )
        except Exception as e:
            return _fail(f"[错误] 读取文件失败: {e}", error_code="EXECUTION_ERROR")


class WriteFileTool(Tool):
    """写入或覆盖文件内容."""

    name = "write_file"
    requires_approval = True
    description = (
        "向指定文件写入内容. 如果文件已存在, 将覆盖原有内容. "
        "用于创建新文件或完全重写文件. 这也是完成任务的核心工具之一, "
        "当你需要创建新文件或重写整个文件时, 应立即使用."
    )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("path", "string", "要写入的文件路径"),
            ToolParameter("content", "string", "要写入的文件内容"),
        ]

    def execute(self, path: str, content: str) -> ToolResult:  # type: ignore[override]
        ok, result = _resolve_path_or_error(self, path, must_exist=False)
        if not ok:
            return _fail(str(result), error_code="PATH_BOUNDARY_ERROR")
        path = str(result)

        try:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            return _ok(f"[成功] 文件已写入: {path} ({len(content)} 字符)")
        except Exception as e:
            return _fail(f"[错误] 写入文件失败: {e}", error_code="EXECUTION_ERROR")


def _resolve_path_or_error(
    tool: Tool, path: str, must_exist: bool = False
) -> Tuple[bool, Union[Path, str]]:
    """解析路径，失败时返回错误字符串.

    Returns:
        (success, result): success 为 True 时 result 为解析后的 Path；
        为 False 时 result 为错误消息字符串。
    """
    try:
        return True, tool._resolve_path(path, must_exist=must_exist)
    except Exception as e:
        return False, f"[错误] {e}"


def _find_best_match(content: str, old_string: str) -> tuple:
    """寻找最接近 old_string 的匹配片段.

    基于 difflib.SequenceMatcher 在 content 中滑动窗口搜索，
    支持行数相近的候选（old_len ± 1）。
    """
    if not old_string.strip():
        return None, 0.0

    if old_string in content:
        return old_string, 1.0

    content_lines = content.splitlines()
    old_lines = old_string.splitlines()
    old_len = len(old_lines)

    if old_len == 0:
        return None, 0.0

    best_match = None
    best_ratio = 0.0

    for i in range(len(content_lines) - old_len + 1):
        candidate = "\n".join(content_lines[i : i + old_len])
        ratio = difflib.SequenceMatcher(None, old_string, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = candidate

    for window in [old_len - 1, old_len + 1]:
        if window <= 0:
            continue
        for i in range(len(content_lines) - window + 1):
            candidate = "\n".join(content_lines[i : i + window])
            ratio = difflib.SequenceMatcher(None, old_string, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = candidate

    return best_match, best_ratio


@dataclass
class SearchReplaceBlock:
    """一个 SEARCH/REPLACE 编辑块."""

    search: str
    replace: str
    raw: str


class SearchReplaceError(ValueError):
    """SEARCH/REPLACE 编辑块解析或校验错误."""


def _parse_search_replace_blocks(text: str) -> List[SearchReplaceBlock]:
    """解析文本中的 SEARCH/REPLACE 编辑块.

    支持的格式（每个块独立，一个字符串可包含多个块）:

        <<<<<<< SEARCH
        要被替换的旧内容
        =======
        替换后的新内容
        >>>>>>> REPLACE

    每个分隔符必须单独占一行。保留 search / replace 中的原始换行符与缩进。
    """
    blocks: List[SearchReplaceBlock] = []
    # splitlines(keepends=True) 保留 \n 或 \r\n，确保替换时与文件内容精确匹配
    lines = text.splitlines(keepends=True)
    i = 0
    n = len(lines)

    while i < n:
        # 查找块起始标记
        while i < n and lines[i].rstrip("\r\n") != "<<<<<<< SEARCH":
            i += 1
        if i >= n:
            break
        start_line = i
        i += 1

        # 收集 search 内容
        search_lines: List[str] = []
        while i < n and lines[i].rstrip("\r\n") != "=======":
            search_lines.append(lines[i])
            i += 1
        if i >= n:
            raise SearchReplaceError("编辑块缺少 '=======' 分隔符")
        i += 1  # 跳过 =======

        # 收集 replace 内容
        replace_lines: List[str] = []
        while i < n and lines[i].rstrip("\r\n") != ">>>>>>> REPLACE":
            replace_lines.append(lines[i])
            i += 1
        if i >= n:
            raise SearchReplaceError("编辑块缺少 '>>>>>>> REPLACE' 结束符")
        i += 1  # 跳过 >>>>>>> REPLACE

        search = "".join(search_lines)
        replace = "".join(replace_lines)
        raw = "".join(lines[start_line:i])
        blocks.append(SearchReplaceBlock(search=search, replace=replace, raw=raw))

    if not blocks:
        raise SearchReplaceError("未找到任何 SEARCH/REPLACE 编辑块")

    return blocks


class SearchReplaceTool(Tool):
    """使用 SEARCH/REPLACE 编辑块安全地修改文件."""

    name = "edit_file_blocks"
    requires_approval = True
    description = (
        "使用 SEARCH/REPLACE 编辑块修改现有文件. 这是修改现有文件时最推荐的方式.\n"
        "每个编辑块包含一段必须在文件中唯一出现的旧内容，以及替换后的新内容.\n"
        "一个调用可包含多个编辑块，系统会先校验所有块，再原子性地写入文件.\n"
        "编辑块格式如下（每个块以 <<<<<<< SEARCH 开始，>>>>>>> REPLACE 结束）:\n"
        "<<<<<<< SEARCH\n"
        "要替换的旧内容（必须在文件中唯一出现）\n"
        "=======\n"
        "替换后的新内容\n"
        ">>>>>>> REPLACE\n"
        "如果旧内容不存在或不唯一，系统会返回详细错误并提示修正."
    )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("path", "string", "要编辑的文件路径"),
            ToolParameter(
                "blocks",
                "string",
                "一个或多个 SEARCH/REPLACE 编辑块，按顺序依次应用",
            ),
        ]

    def execute(self, path: str, blocks: str) -> ToolResult:  # type: ignore[override]
        ok, result = _resolve_path_or_error(self, path, must_exist=True)
        if not ok:
            return _fail(str(result), error_code="PATH_BOUNDARY_ERROR")
        path = str(result)

        if not os.path.exists(path):
            return _fail(f"[错误] 文件不存在: {path}", error_code="FILE_NOT_FOUND")

        try:
            parsed_blocks = _parse_search_replace_blocks(blocks)
        except SearchReplaceError as e:
            return _fail(f"[错误] 编辑块格式错误: {e}", error_code="VALIDATION_ERROR")

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return _fail(f"[错误] 读取文件失败: {e}", error_code="EXECUTION_ERROR")

        # 校验每个 search 在文件中是否唯一存在
        errors: List[str] = []
        for idx, block in enumerate(parsed_blocks, start=1):
            if not block.search:
                errors.append(f"块 {idx}: SEARCH 内容不能为空")
                continue
            count = content.count(block.search)
            if count == 0:
                best_match, ratio = _find_best_match(content, block.search)
                msg = f"块 {idx}: 未找到精确匹配内容"
                if best_match and ratio > 0.3:
                    msg += f"（系统找到最接近的匹配，相似度 {ratio * 100:.1f}%）"
                errors.append(msg)
            elif count > 1:
                errors.append(
                    f"块 {idx}: 匹配内容不唯一，共出现 {count} 次，请增加上下文使其唯一"
                )

        if errors:
            detail = "\n".join(f"  - {e}" for e in errors)
            return _fail(
                f"[错误] 无法应用编辑块（文件: {path}）:\n{detail}",
                error_code="VALIDATION_ERROR",
            )

        # 按顺序应用所有编辑块
        new_content = content
        for block in parsed_blocks:
            new_content = new_content.replace(block.search, block.replace, 1)

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
        except Exception as e:
            return _fail(f"[错误] 写入文件失败: {e}", error_code="EXECUTION_ERROR")

        return _ok(
            f"[成功] 文件已编辑: {path}（应用 {len(parsed_blocks)} 个编辑块，"
            f"原 {len(content)} 字符 → 新 {len(new_content)} 字符）"
        )


class GrepTool(Tool):
    """在代码库中搜索文本模式."""

    name = "grep"
    description = (
        "在代码库中按正则表达式搜索文本. 用于快速定位函数、变量、字符串等出现的位置.\n"
        "自动排除 .git/, node_modules/, vendor/ 等目录.\n"
        "最多返回 50 条匹配; 结果过多时请缩小 pattern 或加 glob 过滤."
    )

    DEFAULT_EXCLUDE_DIRS = {
        ".git",
        ".github",
        "node_modules",
        "vendor",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        "build",
        "dist",
        ".pytest_cache",
        ".mypy_cache",
        "target",
    }

    DEFAULT_EXCLUDE_GLOBS = {
        "*.min.js",
        "*.min.css",
        "*.map",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.gif",
        "*.ico",
        "*.pdf",
        "*.zip",
        "*.tar",
        "*.gz",
        "*.rar",
        "*.exe",
        "*.dll",
        "*.so",
        "*.dylib",
        "*.wasm",
        "*.woff",
        "*.woff2",
        "*.ttf",
    }

    MAX_RESULTS = 50
    MAX_LINE_LEN = 500

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("pattern", "string", "搜索模式(Python 正则表达式)"),
            ToolParameter(
                "path", "string", "搜索路径, 默认当前目录", required=False, default="."
            ),
            ToolParameter(
                "glob", "string", "文件过滤器, 如 '*.go' 或 '*.{py,js}'", required=False
            ),
            ToolParameter(
                "output_mode",
                "string",
                "content(显示匹配行) 或 files(只显示文件列表)",
                required=False,
                default="content",
            ),
        ]

    def execute(  # type: ignore[override]
        self,
        pattern: str,
        path: str = ".",
        glob: Optional[str] = None,
        output_mode: str = "content",
    ) -> ToolResult:
        ok, result = _resolve_path_or_error(self, path, must_exist=True)
        if not ok:
            return _fail(str(result), error_code="PATH_BOUNDARY_ERROR")
        path = str(result)

        if not os.path.exists(path):
            return _fail(f"[错误] 路径不存在: {path}", error_code="FILE_NOT_FOUND")

        if not os.path.isdir(path):
            return self._search_file(path, pattern, output_mode)

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return _fail(f"[错误] 正则表达式无效: {e}", error_code="VALIDATION_ERROR")

        globs = self._parse_glob(glob)
        results = []
        files_matched = set()

        for root, dirs, files in os.walk(path):
            dirs[:] = [
                d
                for d in dirs
                if d not in self.DEFAULT_EXCLUDE_DIRS and not d.startswith(".")
            ]

            for filename in files:
                if any(
                    fnmatch.fnmatch(filename, g) for g in self.DEFAULT_EXCLUDE_GLOBS
                ):
                    continue
                if filename.startswith("."):
                    continue
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

            if (output_mode == "files" and len(files_matched) >= self.MAX_RESULTS) or (
                output_mode != "files" and len(results) >= self.MAX_RESULTS
            ):
                break

        if output_mode == "files":
            if not files_matched:
                return _ok(f"未找到匹配文件 (pattern={pattern!r})")
            lines = sorted(f.replace(os.sep, "/") for f in files_matched)
            if len(lines) >= self.MAX_RESULTS:
                lines.append(
                    f"... 结果超过 {self.MAX_RESULTS} 条, 已截断。请缩小搜索范围。"
                )
            return _ok("匹配文件:\n" + "\n".join(lines))

        if not results:
            return _ok(f"未找到匹配 (pattern={pattern!r})")

        lines = []
        for filepath, line_no, line_text in results[: self.MAX_RESULTS]:
            rel_path = os.path.relpath(filepath, path).replace(os.sep, "/")
            if len(line_text) > self.MAX_LINE_LEN:
                line_text = line_text[: self.MAX_LINE_LEN] + " ..."
            lines.append(f"{rel_path}:{line_no} | {line_text}")

        if len(results) >= self.MAX_RESULTS:
            lines.append(
                f"... 结果超过 {self.MAX_RESULTS} 条, 已截断。请缩小 pattern 或加 glob 过滤。"
            )

        return _ok("\n".join(lines))

    def _parse_glob(self, glob_str: Optional[str] = None) -> List[str]:
        """解析 glob 字符串. 支持 '{a,b,c}' 语法."""
        if not glob_str:
            return []
        glob_str = glob_str.strip()
        if glob_str.startswith("{") and glob_str.endswith("}"):
            return [g.strip() for g in glob_str[1:-1].split(",")]
        return [glob_str]

    def _search_file(self, filepath: str, pattern: str, output_mode: str) -> ToolResult:
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return _fail(f"[错误] 正则表达式无效: {e}", error_code="VALIDATION_ERROR")
        results = self._search_file_lines(filepath, regex, output_mode)
        if output_mode == "files":
            return _ok(filepath if results else "")
        if not results:
            return _ok(f"未找到匹配 (pattern={pattern!r})")
        lines = []
        for _, line_no, line_text in results:
            if len(line_text) > self.MAX_LINE_LEN:
                line_text = line_text[: self.MAX_LINE_LEN] + " ..."
            lines.append(f"{filepath.replace(os.sep, '/')}:{line_no} | {line_text}")
        return _ok("\n".join(lines))

    def _search_file_lines(
        self, filepath: str, regex: re.Pattern, output_mode: str
    ) -> List:
        results = []
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f, start=1):
                    if regex.search(line):
                        results.append((filepath, line_no, line.rstrip("\n")))
                        if output_mode != "files" and len(results) >= self.MAX_RESULTS:
                            break
        except (UnicodeDecodeError, OSError):
            pass
        return results


class ListDirTool(Tool):
    """列出目录内容."""

    name = "list_dir"
    description = (
        "列出指定目录下的文件和子目录. 用于了解项目结构或查看某个目录中包含的文件."
    )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "path",
                "string",
                "要列出的目录路径, 默认为当前目录",
                required=False,
                default=".",
            ),
        ]

    def execute(self, path: str = ".") -> ToolResult:  # type: ignore[override]
        ok, result = _resolve_path_or_error(self, path, must_exist=True)
        if not ok:
            return _fail(str(result), error_code="PATH_BOUNDARY_ERROR")
        path = str(result)

        if not os.path.exists(path):
            return _fail(f"[错误] 目录不存在: {path}", error_code="FILE_NOT_FOUND")

        if not os.path.isdir(path):
            return _fail(
                f"[错误] '{path}' 不是目录, 是一个文件.",
                error_code="FILE_NOT_FOUND",
            )

        try:
            items = os.listdir(path)
            if not items:
                return _ok(f"目录 '{path}' 为空.")

            dirs = []
            files = []
            for item in sorted(items):
                full = os.path.join(path, item)
                if os.path.isdir(full):
                    dirs.append(f"[D] {item}/")
                else:
                    size = os.path.getsize(full)
                    files.append(f"[F] {item} ({self._format_size(size)})")

            result = f"目录: {os.path.abspath(path)}\n{'='*50}\n"
            result += "\n".join(dirs + files)
            result += f"\n{'='*50}\n共 {len(dirs)} 个目录, {len(files)} 个文件"
            return _ok(result)
        except Exception as e:
            return _fail(f"[错误] 列出目录失败: {e}", error_code="EXECUTION_ERROR")

    @staticmethod
    def _format_size(size: float) -> str:
        size_val = float(size)
        for unit in ["B", "KB", "MB"]:
            if size_val < 1024:
                return f"{size_val:.1f} {unit}"
            size_val /= 1024
        return f"{size_val:.1f} GB"


class GlobTool(Tool):
    """按 glob 模式匹配文件."""

    name = "glob"
    description = (
        "按 glob 模式（pattern）在指定目录（path，默认工作目录）中匹配文件，"
        "结果按修改时间倒序排列，最多返回 1000 条。\n"
        "纯通配符模式（如 **）和含花括号扩展（{a,b,c}）的模式会被拒绝。\n"
        "常用示例: '*.py' 查找当前目录所有 py 文件; 'src/**/*.js' 查找 src 下所有 js 文件。"
    )

    MAX_RESULTS = 1000

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "pattern", "string", "glob 匹配模式，如 '*.py' 或 'src/**/*.js'"
            ),
            ToolParameter(
                "path",
                "string",
                "搜索起始目录，默认当前目录",
                required=False,
                default=".",
            ),
        ]

    def execute(  # type: ignore[override]
        self, pattern: str, path: str = "."
    ) -> ToolResult:
        ok, result = _resolve_path_or_error(self, path, must_exist=True)
        if not ok:
            return _fail(str(result), error_code="PATH_BOUNDARY_ERROR")
        path = str(result)

        if not os.path.exists(path):
            return _fail(f"[错误] 路径不存在: {path}", error_code="FILE_NOT_FOUND")
        if not os.path.isdir(path):
            return _fail(f"[错误] '{path}' 不是目录", error_code="FILE_NOT_FOUND")

        clean_pattern = pattern.strip()

        # 安全检查：拒绝纯通配符模式
        if clean_pattern in ("**", "*", "**/*", "*/**"):
            return _fail(
                "[错误] 纯通配符模式被拒绝，请使用更具体的模式。\n"
                "示例: '*.py'、'src/**/*.js'、'test_*.py'",
                error_code="VALIDATION_ERROR",
            )

        # 安全检查：拒绝花括号扩展
        if "{" in clean_pattern and "}" in clean_pattern:
            return _fail(
                "[错误] 含花括号扩展（{a,b,c}）的模式被拒绝，请展开后分别查询。\n"
                "示例: 将 '*.{py,js}' 拆分为两次查询 '*.py' 和 '*.js'",
                error_code="VALIDATION_ERROR",
            )

        # 拒绝包含 .. 的 glob 模式
        if ".." in clean_pattern:
            return _fail(
                "[错误] glob 模式包含 '..'，被拒绝",
                error_code="VALIDATION_ERROR",
            )

        search_path = os.path.join(path, clean_pattern)
        try:
            matches = glob.glob(search_path, recursive=True)
        except Exception as e:
            return _fail(f"[错误] glob 匹配失败: {e}", error_code="EXECUTION_ERROR")

        # 过滤：只保留文件
        files = [p for p in matches if os.path.isfile(p)]

        if not files:
            return _ok(f"未找到匹配文件 (pattern={clean_pattern!r}, path={path!r})")

        # 按修改时间倒序排列
        files_with_mtime = []
        for f in files:
            try:
                mtime = os.path.getmtime(f)
                files_with_mtime.append((f, mtime))
            except OSError:
                continue

        files_with_mtime.sort(key=lambda x: x[1], reverse=True)

        truncated = False
        if len(files_with_mtime) > self.MAX_RESULTS:
            files_with_mtime = files_with_mtime[: self.MAX_RESULTS]
            truncated = True

        lines = []
        for f, mtime in files_with_mtime:
            rel = os.path.relpath(f, path).replace(os.sep, "/")
            time_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"{rel}  ({time_str})")

        if truncated:
            lines.append(
                f"... 结果超过 {self.MAX_RESULTS} 条，已截断。请缩小 pattern 范围。"
            )

        return _ok(f"匹配文件 ({len(files_with_mtime)} 个):\n" + "\n".join(lines))
