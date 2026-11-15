import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, AsyncGenerator

import httpx
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

from pp.config import Config
from pp.domain import Entry, LLMEvent


class BaseLLM(ABC):
    """The LLM interface: generate() and close(). Adapters convert entries and tools to their provider's format."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._retries = cfg.retries

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    def generate(
        self,
        entries: Sequence[Entry],
        tools: list[dict[str, Any]] | None,
        system_prompt: str | None = None,
        stream: bool = True,
    ) -> AsyncGenerator[LLMEvent, None]: ...

    # ======= Helpers for adapters ======= #

    def _new_client(self) -> AsyncOpenAI:
        # Connect fast or fail fast. The read timeout covers the time between parts of a streamed answer.
        timeout = httpx.Timeout(self.cfg.model.timeout, connect=min(15.0, self.cfg.model.timeout))
        # max_retries=0: _with_retries() is the only retry loop. The SDK would otherwise add its own retries.
        return AsyncOpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key, timeout=timeout, max_retries=0)

    async def _with_retries(self, attempt: Callable[[], AsyncIterator[LLMEvent]]) -> AsyncGenerator[LLMEvent, None]:
        """Run attempt() and yield its events. Retry on retryable API errors with exponential backoff.

        After the first event reaches the caller, there is no retry: a retry would send the same text again.
        A final failure becomes one error event.
        """
        retries = max(1, self._retries)
        for attempt_no in range(retries):
            yielded = False
            try:
                async for event in attempt():
                    yielded = True
                    yield event
                return
            except APIError as e:
                if yielded or attempt_no == retries - 1 or not self._is_retryable(e):
                    yield LLMEvent.stream_error(error=self._error_message(e))
                    return
                await asyncio.sleep(2**attempt_no)

    @staticmethod
    def _is_retryable(error: APIError) -> bool:
        # A timeout already waited the full model.timeout. A retry would make the user wait that long again.
        if isinstance(error, APITimeoutError):
            return False
        if isinstance(error, (RateLimitError, APIConnectionError)):
            return True
        if isinstance(error, APIStatusError):
            # A 4xx error (bad request, auth, not found) does not change on a retry.
            return error.status_code >= 500 or error.status_code in (408, 409)
        return False

    def _error_message(self, error: APIError) -> str:
        if isinstance(error, APITimeoutError):
            return (
                f"The model sent no data for {self.cfg.model.timeout:g} seconds. "
                "To wait longer, increase model.timeout (/config model.timeout 600)."
            )
        if isinstance(error, RateLimitError):
            return f"Rate limit exceeded. {error}"
        if isinstance(error, APIConnectionError):
            return f"API connection error. {error}"
        if isinstance(error, APIStatusError):
            return f"API error (HTTP {error.status_code}). {error}"
        return f"Error decoding response. {error}"

    def _parse_tool_call_args(self, args_str: str) -> dict[str, Any]:
        if not args_str:
            return {}

        try:
            return json.loads(args_str)
        except json.JSONDecodeError:
            return {"raw_args": args_str}
