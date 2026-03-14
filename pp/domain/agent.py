from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Any

from pp.domain.shared import TokenUsage


class AgentEventType(str, Enum):
    # lifecycle
    Start = "start"
    Done = "done"
    Error = "error"
    
    # streaming
    TextDelta = "text_delta"
    TextComplete = "text_complete"


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
