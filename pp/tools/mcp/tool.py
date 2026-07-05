from __future__ import annotations

from typing import Any

from pp.config.config import Config
from pp.domain import ToolInvocation, ToolResult, ToolType
from pp.tools.base import Tool


class MCPTool(Tool):
    def __init__(
        self,
        name: str,
        description: str,
        schema: dict[str, Any],
        client_name: str,
        original_tool_name: str,
        manager: Any,
        config: Config,
    ) -> None:
        super().__init__(config)
        self.name = name
        self.description = description
        # MCP uses standard JSON Schema definitions for parameters.
        # By passing a dictionary to self.schema, to_openai_schema() in base.py
        # will correctly extract and build parameters for OpenAI format.
        self.schema = schema
        self.type = ToolType.MCP
        self.client_name = client_name
        self.original_tool_name = original_tool_name
        self.manager = manager

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        return await self.manager.call_tool(
            self.client_name,
            self.original_tool_name,
            invocation.params,
        )
