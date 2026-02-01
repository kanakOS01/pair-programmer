from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


@dataclass
class LLMConfig:
    model: str
    api_key: str
    base_url: str


@dataclass
class EventType(str, Enum):
    TextDelta = "text_delta"
    Error = "error"
    Done = "done"


@dataclass
class TextDelta:
    text: str

    def __str__(self) -> str:
        return self.text


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    
    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


@dataclass
class StreamEvent:
    type: EventType
    text_delta: TextDelta | None = None
    error: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None
