from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ToolType(str, Enum):
    READ = "read"
    WRITE = "write"
    SHELL = "shell"
    NETWORK = "network"
    MEMORY = "memory"
    MCP = "mcp"


@dataclass
class ToolInvocation:
    cwd: Path
    params: dict[str, Any]


@dataclass
class ToolResult:
    ok: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolConfirmation:
    tool_name: str
    params: dict[str, Any]
    description: str
