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
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def error_result(cls, error: str, output: str = "", **kwargs: Any):
        return cls(
            ok=False,
            output=output,
            error=error,
        )

    @classmethod
    def success_result(cls, output: str, **kwargs: Any):
        return cls(ok=True, output=output, **kwargs)

    def to_model_output(self) -> str:
        if self.ok:
            return self.output

        return f"Error: {self.error}\n\nOutput:\n{self.output}"


@dataclass
class ToolConfirmation:
    tool_name: str
    params: dict[str, Any]
    description: str
