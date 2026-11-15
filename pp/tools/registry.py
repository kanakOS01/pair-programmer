from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pp.config import Config
from pp.tools.base import Tool
from pp.tools.builtin import get_builtin_tools

if TYPE_CHECKING:
    from pp.core.session import Session

logger = logging.getLogger(__name__)


def create_default_registry(config: Config) -> ToolRegistry:
    registry = ToolRegistry()

    for tool_cls in get_builtin_tools():
        if config.allowed_tools is None or tool_cls.name in config.allowed_tools:
            registry.register(tool_cls(config))

    from pp.tools.subagent import SubagentTool, get_default_subagent_definitions

    for subagent_config in get_default_subagent_definitions():
        tool_name = f"subagent_{subagent_config.name}"
        if config.allowed_tools is None or tool_name in config.allowed_tools:
            registry.register(SubagentTool(subagent_config.name, subagent_config, config))

    return registry


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self.session: Session | None = None

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            logger.warning(f"Overwriting existing tool {tool.name}")

        self._tools[tool.name] = tool
        if self.session is not None:
            tool.session = self.session
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
