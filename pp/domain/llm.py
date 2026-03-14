from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from pp.domain.shared import TextDelta, TokenUsage


@dataclass
class LLMConfig:
    model: str
    api_key: str
    base_url: str
    retries: Optional[int] = 3


class LLMEventType(str, Enum):
    TextDelta = "text_delta"
    Error = "error"
    Done = "done"


@dataclass
class LLMEvent:
    type: LLMEventType
    text_delta: TextDelta | None = None
    error: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None

    @classmethod
    def stream_error(cls, error: str) -> LLMEvent:
        return LLMEvent(
            type=LLMEventType.Error,
            error=error,
        )

    @classmethod
    def stream_done(cls, finish_reason: str | None, usage: TokenUsage | None, text_delta: TextDelta | None = None) -> LLMEvent:
        return LLMEvent(
            type=LLMEventType.Done,
            finish_reason=finish_reason,
            usage=usage,
            text_delta=text_delta
        )

    @classmethod
    def stream_text_delta(cls, text_delta: TextDelta) -> LLMEvent:
        return LLMEvent(
            type=LLMEventType.TextDelta,
            text_delta=text_delta,
        )
