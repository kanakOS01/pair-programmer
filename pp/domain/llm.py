from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pp.domain.shared import TextDelta, TokenUsage


@dataclass
class LLMConfig:
    model: str
    api_key: str
    base_url: str
    retries: int | None = 3


class LLMEventType(str, Enum):
    TextDelta = "text_delta"
    Error = "error"
    Done = "done"

    ToolCallStart = "tool_call_start"
    ToolCallDelta = "tool_call_delta"
    ToolCallDone = "tool_call_done"
    ToolCallExecuted = "tool_call_executed"


@dataclass
class LLMEvent:
    type: LLMEventType
    text_delta: TextDelta | None = None
    error: str | None = None
    finish_reason: str | None = None
    tool_call: ToolCall | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_delta: ToolCallDelta | None = None
    usage: TokenUsage | None = None

    @classmethod
    def stream_error(cls, error: str) -> LLMEvent:
        return LLMEvent(
            type=LLMEventType.Error,
            error=error,
        )

    @classmethod
    def stream_done(
        cls,
        finish_reason: str | None,
        usage: TokenUsage | None,
        text_delta: TextDelta | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> LLMEvent:
        return LLMEvent(
            type=LLMEventType.Done, finish_reason=finish_reason, usage=usage, text_delta=text_delta, tool_calls=tool_calls
        )

    @classmethod
    def stream_text_delta(cls, text_delta: TextDelta) -> LLMEvent:
        return LLMEvent(
            type=LLMEventType.TextDelta,
            text_delta=text_delta,
        )


@dataclass
class ToolCall:
    call_id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallDelta:
    call_id: str
    name: str | None = None
    args_delta: str = ""
