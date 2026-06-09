"""文件系统相关工具.

提供文件读取、写入、编辑、搜索、glob 匹配等基础操作.
"""

import difflib
import fnmatch
import glob
import os
import re
from datetime import datetime
from typing import List

from ai_coding.tools.base import Tool, ToolParameter


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
            ToolParameter("line_offset", "integer", "起始行号(从1开始), 默认1", required=False, default=1),
            ToolParameter("n_lines", "integer", "读取行数, 默认300, 最大300", required=False, default=300),
        ]

    def execute(self, path: str, line_offset: int = 1, n_lines: int = 300) -> str:
        if not os.path.exists(path):
            return f"[错误] 文件不存在: {path}"

        if os.path.isdir(path):
            return f"[错误] '{path}' 是一个目录, 请使用 list_dir 工具查看目录内容."

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
                f"{start_idx + i + 1:4d} | {line}" for i, line in enumerate(truncated_lines)
            )

            result = f"文件: {path}\n{'='*50}\n{numbered}\n{'='*50}\n"
            result += f"(本段 {start_idx + 1}-{min(end_idx, total_lines)} / 共 {total_lines} 行"
            if total_lines > 1000:
                result += f", 超过 1000 行, 可用 line_offset={min(end_idx + 1, total_lines)} 继续读取"
            result += ")"
            return result

        except UnicodeDecodeError:
            return f"[错误] 无法以文本格式读取 '{path}', 可能是二进制文件."
        except Exception as e:
            return f"[错误] 读取文件失败: {e}"


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

    def execute(self, path: str, content: str) -> str:
        try:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            return f"[成功] 文件已写入: {path} ({len(content)} 字符)"
        except Exception as e:
            return f"[错误] 写入文件失败: {e}"


class EditFile(Tool):
    """在文件中查找并替换指定内容."""

    name = "edit_file"
    requires_approval = True
    description = (
        "在文件中查找并替换指定内容块. 这是修改现有文件的核心工具.\n"
        "old_string 必须在文件中**唯一出现**；如果不唯一，系统会提示你增加上下文.\n"
        "当你已经确认要修改的内容后，立即使用此工具执行修改，不要继续阅读文件."
    )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("path", "string", "要编辑的文件路径"),
            ToolParameter("old_string", "string", "要被替换的旧内容（必须在文件中唯一出现）"),
            ToolParameter("new_string", "string", "用于替换的新内容"),
        ]

    @staticmethod
    def _find_best_match(content: str, old_string: str) -> tuple:
        """寻找最接近的匹配片段."""
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
            candidate = "\n".join(content_lines[i:i + old_len])
            ratio = difflib.SequenceMatcher(None, old_string, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = candidate

        for window in [old_len - 1, old_len + 1]:
            if window <= 0:
                continue
            for i in range(len(content_lines) - window + 1):
                candidate = "\n".join(content_lines[i:i + window])
                ratio = difflib.SequenceMatcher(None, old_string, candidate).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = candidate

        return best_match, best_ratio

    def execute(self, path: str, old_string: str, new_string: str) -> str:
        if not os.path.exists(path):
            return f"[错误] 文件不存在: {path}"

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            if old_string not in content:
                best_match, ratio = self._find_best_match(content, old_string)
                msg = f"[错误] 在文件 '{path}' 中未找到精确匹配内容.\n\n"
                if best_match and ratio > 0.3:
                    msg += f"系统找到最接近的匹配（相似度 {ratio*100:.1f}%）:\n"
                    msg += f"{'-'*40}\n{best_match}\n{'-'*40}\n\n"
                    msg += "请用上述精确文本作为 old_string 重试."
                else:
                    msg += "请重新确认 old_string 与文件内容完全一致（包括缩进和换行）."
                return msg

            occurrences = content.count(old_string)
            if occurrences > 1:
                contexts = []
                idx = 0
                for _ in range(min(occurrences, 3)):
                    idx = content.find(old_string, idx)
                    start = max(0, idx - 50)
                    end = min(len(content), idx + len(old_string) + 50)
                    contexts.append(content[start:end])
                    idx += 1
                msg = f"[错误] old_string 在文件中不唯一，共出现 {occurrences} 次.\n\n"
                msg += "找到的位置:\n"
                for i, ctx in enumerate(contexts, 1):
                    msg += f"--- 位置 {i} ---\n{ctx}\n"
                msg += "\n请增加更多上下文使 old_string 唯一后重试."
                return msg

            new_content = content.replace(old_string, new_string, 1)

            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return f"[成功] 文件已编辑: {path}"
        except Exception as e:
            return f"[错误] 编辑文件失败: {e}"


class GrepTool(Tool):
    """在代码库中搜索文本模式."""

    name = "grep"
    description = (
        "在代码库中按正则表达式搜索文本. 用于快速定位函数、变量、字符串等出现的位置.\n"
        "自动排除 .git/, node_modules/, vendor/ 等目录.\n"
        "最多返回 50 条匹配; 结果过多时请缩小 pattern 或加 glob 过滤."
    )

    DEFAULT_EXCLUDE_DIRS = {
        ".git", ".github", "node_modules", "vendor",
        "__pycache__", ".venv", "venv", ".tox",
        "build", "dist", ".pytest_cache", ".mypy_cache",
        "target",
    }

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
            return self._search_file(path, pattern, output_mode)

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"[错误] 正则表达式无效: {e}"

        globs = self._parse_glob(glob)
        results = []
        files_matched = set()

        for root, dirs, files in os.walk(path):
            dirs[:] = [
                d for d in dirs
                if d not in self.DEFAULT_EXCLUDE_DIRS
                and not d.startswith(".")
            ]

            for filename in files:
                if any(fnmatch.fnmatch(filename, g) for g in self.DEFAULT_EXCLUDE_GLOBS):
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

        if not results:
            return f"未找到匹配 (pattern={pattern!r})"

        lines = []
        for filepath, line_no, line_text in results[:self.MAX_RESULTS]:
            rel_path = os.path.relpath(filepath, path).replace(os.sep, "/")
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
            return [g.strip() for g in glob_str[1:-1].split(",")]
        return [glob_str]

    def _search_file(self, filepath: str, pattern: str, output_mode: str) -> str:
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
    description = "列出指定目录下的文件和子目录. 用于了解项目结构或查看某个目录中包含的文件."

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("path", "string", "要列出的目录路径, 默认为当前目录", required=False, default="."),
        ]

    def execute(self, path: str = ".") -> str:
        if not os.path.exists(path):
            return f"[错误] 目录不存在: {path}"

        if not os.path.isdir(path):
            return f"[错误] '{path}' 不是目录, 是一个文件."

        try:
            items = os.listdir(path)
            if not items:
                return f"目录 '{path}' 为空."

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
            return result
        except Exception as e:
            return f"[错误] 列出目录失败: {e}"

    @staticmethod
    def _format_size(size: int) -> str:
        for unit in ["B", "KB", "MB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GB"


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
            ToolParameter("pattern", "string", "glob 匹配模式，如 '*.py' 或 'src/**/*.js'"),
            ToolParameter("path", "string", "搜索起始目录，默认当前目录", required=False, default="."),
        ]

    def execute(self, pattern: str, path: str = ".") -> str:
        if not os.path.exists(path):
            return f"[错误] 路径不存在: {path}"
        if not os.path.isdir(path):
            return f"[错误] '{path}' 不是目录"

        clean_pattern = pattern.strip()

        # 安全检查：拒绝纯通配符模式
        if clean_pattern in ("**", "*", "**/*", "*/**"):
            return (
                "[错误] 纯通配符模式被拒绝，请使用更具体的模式。\n"
                "示例: '*.py'、'src/**/*.js'、'test_*.py'"
            )

        # 安全检查：拒绝花括号扩展
        if "{" in clean_pattern and "}" in clean_pattern:
            return (
                "[错误] 含花括号扩展（{a,b,c}）的模式被拒绝，请展开后分别查询。\n"
                "示例: 将 '*.{py,js}' 拆分为两次查询 '*.py' 和 '*.js'"
            )

        search_path = os.path.join(path, clean_pattern)
        try:
            matches = glob.glob(search_path, recursive=True)
        except Exception as e:
            return f"[错误] glob 匹配失败: {e}"

        # 过滤：只保留文件
        files = [p for p in matches if os.path.isfile(p)]

        if not files:
            return f"未找到匹配文件 (pattern={clean_pattern!r}, path={path!r})"

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
            files_with_mtime = files_with_mtime[:self.MAX_RESULTS]
            truncated = True

        lines = []
        for f, mtime in files_with_mtime:
            rel = os.path.relpath(f, path).replace(os.sep, "/")
            time_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"{rel}  ({time_str})")

        if truncated:
            lines.append(f"... 结果超过 {self.MAX_RESULTS} 条，已截断。请缩小 pattern 范围。")

        return f"匹配文件 ({len(files_with_mtime)} 个):\n" + "\n".join(lines)
