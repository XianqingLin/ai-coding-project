"""MCP 工具加载器.

读取 mcp_servers.json 配置, 连接外部 MCP Server,
并将暴露的工具转换为内部 Tool 实例.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from ai_coding.config import get_env, get_env_bool
from ai_coding.logger import get_logger
from ai_coding.mcp.client import MCPClient
from ai_coding.tools.base import Tool
from ai_coding.tools.mcp_adapter import MCPToolAdapter

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = "mcp_servers.json"


class MCPLoader:
    """MCP Server 配置加载与工具转换."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        """初始化加载器.

        Args:
            config_path: MCP 配置文件路径, 默认使用项目根目录的
                mcp_servers.json 或环境变量 MCP_SERVERS_CONFIG.
        """
        self.config_path = config_path or get_env(
            "MCP_SERVERS_CONFIG", DEFAULT_CONFIG_PATH
        )
        self.clients: List[MCPClient] = []
        self.tools: List[Tool] = []

    def load(self) -> Tuple[List[Tool], List[MCPClient]]:
        """加载配置、连接 Server、返回工具与 Client 列表.

        Returns:
            (tools, clients). 单个 Server 失败不影响其他 Server.
        """
        if not get_env_bool("MCP_ENABLED", True):
            logger.info("MCP 已禁用, 跳过加载")
            return [], []

        config = self._read_config()
        if config is None:
            return [], []

        servers = config.get("mcpServers", {})
        if not isinstance(servers, dict):
            logger.warning("MCP 配置中 mcpServers 不是字典, 跳过加载")
            return [], []

        for server_name, server_config in servers.items():
            try:
                tools, client = self._connect_server(server_name, server_config)
                self.clients.append(client)
                self.tools.extend(tools)
            except Exception as e:
                logger.warning(f"MCP Server '{server_name}' 加载失败: {e}")

        logger.info(
            f"MCP 共加载 {len(self.tools)} 个工具, 来自 {len(self.clients)} 个 Server"
        )
        return self.tools, self.clients

    def _read_config(self) -> Optional[Dict[str, Any]]:
        """读取配置文件."""
        path = Path(self.config_path)
        if not path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            path = project_root / self.config_path

        if not path.exists():
            logger.debug(f"MCP 配置文件不存在: {path}")
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                return cast(Dict[str, Any], json.load(f))
        except Exception as e:
            logger.warning(f"读取 MCP 配置文件失败: {e}")
            return None

    def _connect_server(
        self, server_name: str, server_config: Dict[str, Any]
    ) -> Tuple[List[Tool], MCPClient]:
        """连接单个 Server 并转换工具."""
        command = server_config.get("command")
        if not command:
            raise ValueError("MCP Server 配置缺少 command")

        args = list(server_config.get("args", []))
        env_config = server_config.get("env") or {}
        env: Optional[Dict[str, str]] = None
        if env_config:
            env = dict(os.environ)
            env.update({str(k): str(v) for k, v in env_config.items()})

        client = MCPClient(
            server_name=server_name,
            command=str(command),
            args=args,
            env=env,
        )
        if not client.connect():
            client.stop()
            raise RuntimeError(f"无法连接 MCP Server '{server_name}'")

        mcp_tools = client.list_tools()
        tools: List[Tool] = []
        for mcp_tool in mcp_tools:
            adapter = MCPToolAdapter(
                server_name=server_name,
                mcp_tool=mcp_tool,
                client=client,
            )
            tools.append(adapter)

        return tools, client

    def disconnect_all(self) -> None:
        """断开所有 MCP Server 连接."""
        for client in self.clients:
            try:
                client.stop()
            except Exception as e:
                logger.debug(f"MCP Client '{client.server_name}' 停止异常: {e}")
        self.clients.clear()
        self.tools.clear()


def load_mcp_tools(
    config_path: Optional[str] = None,
) -> Tuple[List[Tool], List[MCPClient]]:
    """便捷函数: 加载 MCP 工具.

    Args:
        config_path: 配置文件路径, 为 None 时使用默认路径/环境变量.

    Returns:
        (tools, clients).
    """
    loader = MCPLoader(config_path=config_path)
    return loader.load()
