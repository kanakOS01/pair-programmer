from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from pp.domain.shared import TokenUsage
from pp.domain.tools import ToolResult


class AgentEventType(str, Enum):
    # lifecycle
    Start = "start"
    Done = "done"
    Error = "error"

    # streaming
    TextDelta = "text_delta"
    TextComplete = "text_complete"

    # tool call
    ToolCallStart = "tool_call_start"
    ToolCallDone = "tool_call_done"
    ToolCallExecuted = "tool_call_executed"


@dataclass
class AgentEvent:
    type: AgentEventType
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def agent_start(cls, message: str) -> AgentEvent:
        return cls(
            type=AgentEventType.Start,
            data={"message": message},
        )

    @classmethod
    def agent_done(cls, response: str | None = None, usage: TokenUsage | None = None) -> AgentEvent:
        return cls(
            type=AgentEventType.Done,
            data={
                "response": response,
                "usage": asdict(usage) if usage else None,
            },
        )

    @classmethod
    def agent_error(cls, error: str, details: dict[str, Any] | None = None) -> AgentEvent:
        return cls(
            type=AgentEventType.Error,
            data={
                "error": error,
                "details": details,
            },
        )

    @classmethod
    def agent_text_delta(cls, content: str) -> AgentEvent:
        return cls(
            type=AgentEventType.TextDelta,
            data={"content": content},
        )

    @classmethod
    def agent_text_complete(cls, content: str) -> AgentEvent:
        return cls(
            type=AgentEventType.TextComplete,
            data={"content": content},
        )

    @classmethod
    def tool_call_start(cls, call_id: str, name: str | None, args: dict[str, Any]):
        return cls(
            type=AgentEventType.ToolCallStart,
            data={
                "call_id": call_id,
                "name": name,
                "args": args,
            },
        )

    @classmethod
    def tool_call_excecuted(cls, call_id: str, name: str | None, result: ToolResult) -> AgentEvent:
        return cls(
            type=AgentEventType.ToolCallDone,
            data={
                "call_id": call_id,
                "name": name,
                "success": result.ok,
                "error": result.error,
                "output": result.output,
                "metadata": result.metadata,
                "truncated": result.truncated,
            },
        )
