"""文件系统相关工具.

提供文件读取、写入、目录列出、命令执行等基础操作.
"""

import os
import subprocess
from typing import List

from ai_coding.tools.base import Tool, ToolParameter


class ReadFileTool(Tool):
    """读取文件内容."""
    
    name = "read_file"
    description = "读取指定文件的内容, 返回文件中的文本. 用于查看代码、配置文件或文档内容."
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("path", "string", "要读取的文件路径（相对路径或绝对路径）"),
        ]
    
    def execute(self, path: str) -> str:
        if not os.path.exists(path):
            return f"[错误] 文件不存在: {path}"
        
        if os.path.isdir(path):
            return f"[错误] '{path}' 是一个目录, 请使用 list_dir 工具查看目录内容."
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            # 显示行号, 方便引用
            lines = content.split("\n")
            numbered = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
            return f"文件: {path}\n{'='*50}\n{numbered}\n{'='*50}\n(共 {len(lines)} 行)"
        except UnicodeDecodeError:
            return f"[错误] 无法以文本格式读取 '{path}', 可能是二进制文件."
        except Exception as e:
            return f"[错误] 读取文件失败: {e}"


class WriteFileTool(Tool):
    """写入或覆盖文件内容."""
    
    name = "write_file"
    description = "向指定文件写入内容. 如果文件已存在, 将覆盖原有内容. 用于创建新文件或完全重写文件."
    
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


class EditFileTool(Tool):
    """局部编辑文件内容."""
    
    name = "edit_file"
    description = "在文件中查找并替换指定内容. 用于局部修改文件, 而不是完全重写. old_string 必须精确匹配文件中的内容."
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("path", "string", "要编辑的文件路径"),
            ToolParameter("old_string", "string", "要被替换的旧内容（必须精确匹配）"),
            ToolParameter("new_string", "string", "用于替换的新内容"),
        ]
    
    def execute(self, path: str, old_string: str, new_string: str) -> str:
        if not os.path.exists(path):
            return f"[错误] 文件不存在: {path}"
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            if old_string not in content:
                return f"[错误] 在文件 '{path}' 中未找到指定内容, 请确认 old_string 完全匹配."
            
            new_content = content.replace(old_string, new_string, 1)
            
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            return f"[成功] 文件已编辑: {path}"
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


class ExecuteCommandTool(Tool):
    """执行 shell 命令."""
    
    name = "execute_command"
    description = "执行 shell 命令. 用于运行测试、安装依赖、构建项目等操作. 谨慎使用有破坏性的命令."
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("command", "string", "要执行的 shell 命令"),
            ToolParameter("timeout", "integer", "命令超时时间（秒）, 默认 30 秒", required=False, default=30),
        ]
    
    def execute(self, command: str, timeout: int = 30) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            output_parts = []
            
            if result.stdout:
                output_parts.append(f"[stdout]\n{result.stdout}")
            
            if result.stderr:
                output_parts.append(f"[stderr]\n{result.stderr}")
            
            if result.returncode != 0:
                output_parts.append(f"[退出码] {result.returncode}")
            
            if not output_parts:
                return "[成功] 命令执行完成, 无输出."
            
            return "\n\n".join(output_parts)
        
        except subprocess.TimeoutExpired:
            return f"[错误] 命令执行超时（超过 {timeout} 秒）."
        except Exception as e:
            return f"[错误] 命令执行失败: {e}"
