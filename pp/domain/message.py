from __future__ import annotations

from dataclasses import dataclass, field

from pp.domain.llm import ToolCall


@dataclass(frozen=True)
class UserMessage:
    text: str


@dataclass(frozen=True)
class AssistantTurn:
    text: str
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ToolResultEntry:
    call_id: str
    content: str


@dataclass(frozen=True)
class Summary:
    text: str


# One item in the conversation history.
Entry = UserMessage | AssistantTurn | ToolResultEntry | Summary
