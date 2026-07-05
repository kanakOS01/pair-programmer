from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pp.config import Config
from pp.domain import ToolInvocation, ToolResult
from pp.tools.base import Tool
from pp.tools.builtin import get_builtin_tools

logger = logging.getLogger(__name__)


def create_default_registry(config: Config) -> ToolRegistry:
    registry = ToolRegistry()

    for tool_cls in get_builtin_tools():
        if config.allowed_tools is None or tool_cls.name in config.allowed_tools:
            registry.register(tool_cls(config))

    if config.subagents:
        from pp.tools.subagent import SubagentTool

        for name, subagent_config in config.subagents.items():
            registry.register(SubagentTool(name, subagent_config, config))

    return registry


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            logger.warning(f"Overwriting existing tool {tool.name}")

        self._tools[tool.name] = tool
        logger.debug(f"Tool {tool.name} registered")

    def unregister(self, tool: Tool) -> bool:
        if tool.name not in self._tools:
            logger.warning(f"Tool {tool.name} not registered")
            return False

        del self._tools[tool.name]
        return True

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return [tool for tool in self._tools.values()]

    def get_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self.list_tools()]

    async def invoke(
        self,
        name: str,
        params: dict[str, Any],
        cwd: str | Path,
    ) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult.error_result(f"Tool {name} not found", metadata={"tool_name": name})

        validation_errors = tool.validate_params(params)
        if validation_errors:
            return ToolResult.error_result(
                f"Invalid parameters: {'; '.join(validation_errors)}",
                metadata={
                    "tool_name": name,
                    "validate_errors": validation_errors,
                },
            )

        invocation = ToolInvocation(cwd=Path(cwd), params=params)
        try:
            return await tool.execute(invocation)
        except Exception as e:
            logger.exception(f"Tool {name} raised unexpected error")
            return ToolResult.error_result(f"Internal error: {e}", metadata={"tool_name": name})
