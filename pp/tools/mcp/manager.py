from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pp.config.config import Config
from pp.domain import ToolResult
from pp.tools.mcp.client import MCPClient
from pp.tools.mcp.tool import MCPTool

logger = logging.getLogger(__name__)


class MCPManager:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.clients: dict[str, MCPClient] = {}
        # Why each server that failed to start did not start, by server name.
        self.errors: dict[str, str] = {}
        for server_name, server_cfg in config.mcp_servers.items():
            self.clients[server_name] = MCPClient(
                name=server_name,
                command=server_cfg.command,
                args=server_cfg.args,
                env=server_cfg.env,
            )

    async def initialize(self) -> None:
        """Connect all configured MCP servers at the same time. A server that fails is recorded in self.errors."""
        self.errors.clear()
        await asyncio.gather(*(self._connect(name, client) for name, client in self.clients.items()))

    async def _connect(self, name: str, client: MCPClient) -> None:
        timeout = self.config.mcp_servers[name].startup_timeout
        try:
            await client.connect(timeout=timeout)
        except TimeoutError:
            self.errors[name] = (
                f"The server did not answer within {timeout:g} seconds. If it downloads packages when it starts "
                f"(for example with uvx), set startup_timeout in [mcp_servers.{name}] to wait longer."
            )
        except Exception as e:
            self.errors[name] = str(e)
        else:
            return
        # The CLI shows these errors. Log at debug level only, so they are not printed twice.
        logger.debug(f"Failed to connect to MCP server '{name}': {self.errors[name]}")

    async def get_tools(self) -> list[MCPTool]:
        """Query all connected clients for their tools and wrap them in MCPTool instances."""
        tools = []
        for name, client in self.clients.items():
            if not client.connected:
                continue

            try:
                client_tools = await client.list_tools()
                for t in client_tools:
                    # Prefix tool names to guarantee uniqueness across servers and built-in tools
                    tool_name = f"mcp_{name}_{t['name']}"
                    schema = t.get("inputSchema", t.get("schema", {}))

                    mcp_tool = MCPTool(
                        name=tool_name,
                        description=t.get("description", ""),
                        schema=schema,
                        client_name=name,
                        original_tool_name=t["name"],
                        manager=self,
                        config=self.config,
                    )
                    tools.append(mcp_tool)
            except Exception as e:
                logger.error(f"Failed to list tools for MCP server '{name}': {e}")
        return tools

    async def call_tool(self, client_name: str, original_tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        client = self.clients.get(client_name)
        if not client:
            return ToolResult.error_result(f"MCP client '{client_name}' not found")

        try:
            result = await client.call_tool(original_tool_name, arguments)
            is_error = result.get("isError", False)
            content_items = result.get("content", [])

            outputs = []
            for item in content_items:
                if item.get("type") == "text":
                    outputs.append(item.get("text", ""))
                else:
                    outputs.append(json.dumps(item))

            output_text = "\n".join(outputs)

            if is_error:
                return ToolResult.error_result(output_text)
            else:
                return ToolResult.success_result(output_text)
        except Exception as e:
            return ToolResult.error_result(f"MCP tool execution failed: {e}")

    async def close(self) -> None:
        """Disconnect and stop all MCP server processes."""
        for client in self.clients.values():
            try:
                await client.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting MCP client '{client.name}': {e}")
