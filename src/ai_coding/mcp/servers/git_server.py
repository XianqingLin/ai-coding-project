"""本地 Git MCP Server 示例.

通过 MCP 协议暴露简单的 Git 查询工具, 不依赖 Node.js,
可直接用 Python 运行, 方便测试与演示.

运行方式:
    python -m ai_coding.mcp.servers.git_server
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("git-server")


async def _run_git(path: str, args: list[str]) -> str:
    """在指定目录异步运行 git 命令.

    使用 asyncio.to_thread 避免阻塞事件循环,
    并设置 stdin=DEVNULL 防止 git 在 Windows 上等待终端输入.
    """
    cwd = Path(path).expanduser().resolve()
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    stdout, stderr = await proc.communicate()
    text = stdout.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace").strip()
        return f"[git error] {err or 'unknown error'}"
    return text


@mcp.tool()
async def get_git_status(path: str) -> str:
    """获取指定目录的 git status.

    Args:
        path: 目标目录路径.

    Returns:
        git status 文本输出.
    """
    return await _run_git(path, ["status", "--short"])


@mcp.tool()
async def get_git_log(path: str, count: int = 5) -> str:
    """获取指定目录最近的 git 提交日志.

    Args:
        path: 目标目录路径.
        count: 返回的提交数量, 默认 5.

    Returns:
        git log 文本输出.
    """
    return await _run_git(path, ["log", f"-{max(1, count)}", "--oneline"])


@mcp.tool()
async def get_git_branch(path: str) -> str:
    """获取指定目录当前分支.

    Args:
        path: 目标目录路径.

    Returns:
        当前分支名称.
    """
    return await _run_git(path, ["branch", "--show-current"])


def main() -> None:
    """启动 stdio MCP Server."""
    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
