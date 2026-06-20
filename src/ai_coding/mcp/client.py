"""MCP Client 同步封装.

MCP Python SDK 基于 asyncio, 而项目内部工具接口是同步的.
此类在后台线程中运行事件循环, 对外暴露同步方法.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, cast

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult, TextContent, Tool

from ai_coding.logger import get_logger

logger = get_logger(__name__)


class MCPClient:
    """单个 MCP Server 的同步客户端封装.

    负责启动外部 Server 进程、建立 MCP 会话、列出工具并调用工具.
    连接失败会记录 warning, 不会阻塞其他 Server 或内置工具.
    """

    def __init__(
        self,
        server_name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ) -> None:
        """初始化 MCP Client.

        Args:
            server_name: Server 别名, 用于工具名前缀隔离.
            command: 启动 Server 的命令.
            args: 启动 Server 的参数列表.
            env: 额外环境变量.
            timeout: 连接和调用的默认超时(秒).
        """
        self.server_name = server_name
        self.command = command
        self.args = list(args or [])
        self.env = env
        self.timeout = timeout

        self._session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()
        self._connected = False

        # 后台事件循环, 用于承载异步 MCP 会话
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        """在后台线程中运行事件循环."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coro(self, coro: Any) -> Any:
        """将协程提交到后台事件循环并等待结果."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=self.timeout)

    def connect(self) -> bool:
        """连接 MCP Server.

        Returns:
            是否连接成功.
        """
        if self._connected:
            return True
        try:
            self._run_coro(self._connect())
            self._connected = True
            logger.info(f"MCP Server '{self.server_name}' 已连接")
            return True
        except Exception as e:
            logger.warning(f"MCP Server '{self.server_name}' 连接失败: {e}")
            self._connected = False
            return False

    async def _connect(self) -> None:
        """异步建立连接."""
        params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )
        streams = await self._exit_stack.enter_async_context(stdio_client(params))
        read_stream, write_stream = streams
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

    def list_tools(self) -> List[Tool]:
        """获取 Server 暴露的工具列表.

        Returns:
            工具元数据列表. 未连接时返回空列表.
        """
        if not self._connected or self._session is None:
            return []
        try:
            result = self._run_coro(self._session.list_tools())
            return list(result.tools)
        except Exception as e:
            logger.warning(f"MCP Server '{self.server_name}' 列出工具失败: {e}")
            return []

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """调用指定工具.

        Args:
            name: 工具名称(Server 侧原名称).
            arguments: 工具参数.

        Returns:
            MCP 调用结果.

        Raises:
            RuntimeError: 未连接时.
        """
        if not self._connected or self._session is None:
            raise RuntimeError(f"MCP Server '{self.server_name}' 未连接")
        result = self._run_coro(self._session.call_tool(name, arguments))
        return cast(CallToolResult, result)

    def disconnect(self) -> None:
        """断开连接并清理资源."""
        if not self._connected:
            return
        try:
            self._run_coro(self._exit_stack.aclose())
        except Exception as e:
            logger.debug(f"MCP Server '{self.server_name}' 清理时异常: {e}")
        finally:
            self._connected = False
            self._session = None

    def stop(self) -> None:
        """停止后台事件循环."""
        try:
            self.disconnect()
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)


def format_tool_result(result: CallToolResult) -> str:
    """将 MCP CallToolResult 转换为字符串.

    目前只处理 TextContent, 图片内容会提示忽略.

    Args:
        result: MCP 工具调用结果.

    Returns:
        可注入对话的文本结果.
    """
    texts: List[str] = []
    for item in result.content:
        if isinstance(item, TextContent):
            texts.append(item.text)
        else:
            texts.append("[非文本内容, 已忽略]")
    if result.isError:
        return f"[工具调用错误] {' '.join(texts)}"
    return "\n".join(texts)
