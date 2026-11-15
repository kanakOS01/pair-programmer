from typing import Any

import httpx
import pytest
from openai import APIConnectionError, BadRequestError, InternalServerError, RateLimitError

import pp.llm.base as llm_base
from pp.config import Config
from pp.domain import LLMEvent, LLMEventType, TextDelta, UserMessage
from pp.llm import OpenRouterLLM

REQUEST = httpx.Request("POST", "http://test")


def status_error(cls, code: int):
    return cls("error", response=httpx.Response(code, request=REQUEST), body=None)


class ScriptedLLM(OpenRouterLLM):
    """Each attempt yields `before_error` text events, then raises the next error (or finishes)."""

    def __init__(self, errors: list[Exception | None], before_error: int = 0) -> None:
        super().__init__(Config())
        self.errors = list(errors)
        self.before_error = before_error
        self.attempts = 0
        self.params: list[dict[str, Any]] = []

    def get_client(self) -> Any:
        return None

    async def _generate_stream(self, client, kwargs):
        self.attempts += 1
        self.params.append(kwargs)
        for _ in range(self.before_error):
            yield LLMEvent.stream_text_delta(text_delta=TextDelta(text="hi"))
        error = self.errors.pop(0)
        if error:
            raise error
        yield LLMEvent.stream_done(finish_reason="stop", usage=None)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch):
    async def sleep(_):
        pass

    monkeypatch.setattr(llm_base.asyncio, "sleep", sleep)


async def collect(llm: OpenRouterLLM) -> list[LLMEvent]:
    return [event async for event in llm.generate([UserMessage("x")], None)]


@pytest.mark.parametrize(
    "error",
    [
        APIConnectionError(request=REQUEST),
        status_error(RateLimitError, 429),
        status_error(InternalServerError, 500),
    ],
)
async def test_retryable_errors_are_retried(error):
    llm = ScriptedLLM([error, None])

    events = await collect(llm)

    assert llm.attempts == 2
    assert events[-1].type == LLMEventType.Done


async def test_bad_request_is_not_retried():
    llm = ScriptedLLM([status_error(BadRequestError, 400), None])

    events = await collect(llm)

    assert llm.attempts == 1
    assert events[-1].type == LLMEventType.Error
    assert "400" in events[-1].error


async def test_no_retry_after_text_was_sent():
    llm = ScriptedLLM([APIConnectionError(request=REQUEST), None], before_error=1)

    events = await collect(llm)

    assert llm.attempts == 1
    assert [e.type for e in events] == [LLMEventType.TextDelta, LLMEventType.Error]


async def test_error_after_last_retry():
    llm = ScriptedLLM([APIConnectionError(request=REQUEST)] * 3)

    events = await collect(llm)

    assert llm.attempts == 3
    assert events[-1].type == LLMEventType.Error


async def test_temperature_is_sent():
    llm = ScriptedLLM([None])
    llm.cfg.temperature = 0.3

    await collect(llm)

    assert llm.params[0]["temperature"] == 0.3
