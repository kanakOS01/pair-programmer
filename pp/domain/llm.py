from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
from enum import Enum


@dataclass
class LLMConfig:
    model: str
    api_key: str
    base_url: str
    retries: Optional[int] = 3


@dataclass
class LLMEventType(str, Enum):
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
    type: LLMEventType
    text_delta: TextDelta | None = None
    error: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None

    @classmethod
    def stream_error(cls, error: str) -> StreamEvent:
        return StreamEvent(
            type=LLMEventType.Error,
            error=error,
        )

    @classmethod
    def stream_done(cls, finish_reason: str | None, usage: TokenUsage | None, text_delta: TextDelta | None = None) -> StreamEvent:
        return StreamEvent(
            type=LLMEventType.Done,
            finish_reason=finish_reason,
            usage=usage,
            text_delta=text_delta
        )

    @classmethod
    def stream_text_delta(cls, text_delta: TextDelta) -> StreamEvent:
        return StreamEvent(
            type=LLMEventType.TextDelta,
            text_delta=text_delta,
        )
