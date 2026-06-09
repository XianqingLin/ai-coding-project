"""文件系统相关工具.

提供文件读取、写入、目录列出、代码编辑等基础操作.
"""

import difflib
import os
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
            # 确保目录存在
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            return f"[成功] 文件已写入: {path} ({len(content)} 字符)"
        except Exception as e:
            return f"[错误] 写入文件失败: {e}"


class StrReplaceFileTool(Tool):
    """在文件中查找并替换指定内容."""

    name = "str_replace_file"
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
                # 精确匹配失败 → 模糊匹配 fallback
                best_match, ratio = self._find_best_match(content, old_string)
                msg = f"[错误] 在文件 '{path}' 中未找到精确匹配内容.\n\n"
                if best_match and ratio > 0.3:
                    msg += f"系统找到最接近的匹配（相似度 {ratio*100:.1f}%）:\n"
                    msg += f"{'-'*40}\n{best_match}\n{'-'*40}\n\n"
                    msg += "请用上述精确文本作为 old_string 重试."
                else:
                    msg += "请重新确认 old_string 与文件内容完全一致（包括缩进和换行）."
                return msg

            # 唯一性检查
            occurrences = content.count(old_string)
            if occurrences > 1:
                # 找到所有出现位置并显示上下文
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


class InsertAfterLineTool(Tool):
    """在指定行后插入内容."""

    name = "insert_after_line"
    description = (
        "在文件的指定行号之后插入新内容.\n"
        "常用于在函数后添加新函数、在结构体中添加新字段等场景."
    )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("path", "string", "要编辑的文件路径"),
            ToolParameter("line_number", "integer", "在此行号之后插入（从1开始）"),
            ToolParameter("content", "string", "要插入的内容"),
        ]

    def execute(self, path: str, line_number: int, content: str) -> str:
        if not os.path.exists(path):
            return f"[错误] 文件不存在: {path}"

        try:
            line_number = int(line_number)
            if line_number < 0:
                return f"[错误] 行号必须 >= 0"

            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().split("\n")

            total = len(lines)
            insert_idx = line_number  # 在第 line_number 行之后插入

            if insert_idx > total:
                insert_idx = total

            new_lines = lines[:insert_idx] + content.split("\n") + lines[insert_idx:]

            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines))

            return (
                f"[成功] 文件已编辑: {path}\n"
                f"在第 {line_number} 行后插入了 {len(content.split(chr(10)))} 行"
            )
        except Exception as e:
            return f"[错误] 编辑文件失败: {e}"


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

            # 分类显示：目录在前，文件在后
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
        """格式化文件大小."""
        for unit in ["B", "KB", "MB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GB"



