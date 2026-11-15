import json
from types import SimpleNamespace as NS
from typing import Any

import httpx
import pytest
from openai import APIConnectionError

import pp.llm.base as llm_base
from pp.config import Config
from pp.config.config import Provider
from pp.domain import AssistantTurn, LLMEventType, Summary, ToolCall, ToolResultEntry, UserMessage
from pp.llm import OpenAIResponsesLLM, OpenRouterLLM, ProviderRouterLLM
from pp.llm.openai_responses import to_responses_input, to_responses_tools


def usage(input_tokens=10, output_tokens=5, cached=3):
    return NS(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=NS(cached_tokens=cached),
    )


def function_call(call_id: str, name: str, args: dict[str, Any]):
    return NS(type="function_call", call_id=call_id, name=name, arguments=json.dumps(args))


class FakeResponses:
    """Stands in for client.responses. Records the params and plays back events or a response."""

    def __init__(self, events=None, response=None, errors=()):
        self.events = events or []
        self.response = response
        self.errors = list(errors)
        self.params: list[dict[str, Any]] = []

    async def create(self, **params):
        self.params.append(params)
        if self.errors:
            raise self.errors.pop(0)
        if params.get("stream"):
            return self._stream()
        return self.response

    async def _stream(self):
        for event in self.events:
            yield event


def make_llm(config: Config, responses: FakeResponses) -> OpenAIResponsesLLM:
    llm = OpenAIResponsesLLM(config)
    llm._client = NS(responses=responses)  # type: ignore[assignment]
    return llm


async def collect(llm, entries=(UserMessage("hi"),), tools=None, system_prompt=None, stream=True):
    return [e async for e in llm.generate(list(entries), tools, system_prompt=system_prompt, stream=stream)]


# ---------- request building ----------


def test_history_becomes_responses_input():
    entries = [
        Summary("earlier work"),
        UserMessage("read a.py"),
        AssistantTurn("Reading it.", (ToolCall("c1", "read_file", {"path": "a.py"}),)),
        ToolResultEntry("c1", "file body"),
        AssistantTurn("", (ToolCall("c2", "list_dir", {}),)),
        ToolResultEntry("c2", "a.py"),
        AssistantTurn("Done."),
    ]

    assert to_responses_input(entries) == [
        {"role": "developer", "content": "Summary of previous conversation:\nearlier work"},
        {"role": "user", "content": "read a.py"},
        {"role": "assistant", "content": "Reading it."},
        {"type": "function_call", "call_id": "c1", "name": "read_file", "arguments": '{"path": "a.py"}'},
        {"type": "function_call_output", "call_id": "c1", "output": "file body"},
        {"type": "function_call", "call_id": "c2", "name": "list_dir", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c2", "output": "a.py"},
        {"role": "assistant", "content": "Done."},
    ]


def test_tools_become_flat_non_strict_function_tools():
    tools = [{"name": "read_file", "description": "Read", "parameters": {"type": "object", "properties": {}}}]

    assert to_responses_tools(tools) == [
        {
            "type": "function",
            "name": "read_file",
            "description": "Read",
            "parameters": {"type": "object", "properties": {}},
            "strict": False,
        }
    ]


async def test_request_params(config: Config):
    responses = FakeResponses(events=[NS(type="response.completed", response=NS(status="completed", usage=usage()))])
    config.model.reasoning_effort = "high"
    tools = [{"name": "read_file", "description": "Read", "parameters": {"type": "object"}}]

    await collect(make_llm(config, responses), tools=tools, system_prompt="SYS")

    params = responses.params[0]
    assert params["model"] == "gpt-6.1-sol"
    assert params["instructions"] == "SYS"
    assert params["store"] is False
    assert params["stream"] is True
    assert params["reasoning"] == {"effort": "high"}
    assert params["tools"][0]["name"] == "read_file"
    # No temperature unless the user sets one: reasoning models can reject it.
    assert "temperature" not in params


async def test_temperature_is_sent_when_set(config: Config):
    responses = FakeResponses(events=[NS(type="response.completed", response=NS(status="completed", usage=None))])
    config.model.temp = 0.2

    await collect(make_llm(config, responses))

    assert responses.params[0]["temperature"] == 0.2


# ---------- streaming ----------


async def test_stream_text_tool_calls_and_usage(config: Config):
    responses = FakeResponses(
        events=[
            NS(type="response.created"),
            NS(type="response.output_text.delta", delta="Let me "),
            NS(type="response.output_text.delta", delta="look."),
            NS(type="response.output_item.done", item=NS(type="message")),
            NS(type="response.output_item.done", item=function_call("call_1", "read_file", {"path": "a.py"})),
            NS(type="response.output_item.done", item=function_call("call_2", "grep", {"pattern": "x"})),
            NS(type="response.completed", response=NS(status="completed", usage=usage())),
        ]
    )

    events = await collect(make_llm(config, responses))

    text = "".join(e.text_delta.text for e in events if e.type == LLMEventType.TextDelta)
    calls = [e.tool_call for e in events if e.type == LLMEventType.ToolCallDone]
    done = events[-1]
    assert text == "Let me look."
    assert calls == [ToolCall("call_1", "read_file", {"path": "a.py"}), ToolCall("call_2", "grep", {"pattern": "x"})]
    assert done.type == LLMEventType.Done
    assert (done.usage.prompt_tokens, done.usage.completion_tokens, done.usage.cached_tokens) == (10, 5, 3)


async def test_stream_failed_and_error_events(config: Config):
    failed = FakeResponses(events=[NS(type="response.failed", response=NS(error=NS(message="model overloaded")))])
    error = FakeResponses(events=[NS(type="error", message="bad input")])

    failed_events = await collect(make_llm(config, failed))
    error_events = await collect(make_llm(config, error))

    assert failed_events[-1].type == LLMEventType.Error and "model overloaded" in failed_events[-1].error
    assert error_events[-1].type == LLMEventType.Error and "bad input" in error_events[-1].error


async def test_stream_incomplete_is_done_with_reason(config: Config):
    responses = FakeResponses(
        events=[
            NS(type="response.output_text.delta", delta="partial"),
            NS(
                type="response.incomplete",
                response=NS(incomplete_details=NS(reason="max_output_tokens"), usage=usage()),
            ),
        ]
    )

    events = await collect(make_llm(config, responses))

    assert events[-1].type == LLMEventType.Done
    assert events[-1].finish_reason == "incomplete: max_output_tokens"


async def test_bad_tool_arguments_are_kept_raw(config: Config):
    item = NS(type="function_call", call_id="c", name="shell", arguments="{not json")
    responses = FakeResponses(
        events=[
            NS(type="response.output_item.done", item=item),
            NS(type="response.completed", response=NS(status="completed", usage=None)),
        ]
    )

    events = await collect(make_llm(config, responses))

    assert events[0].tool_call.args == {"raw_args": "{not json"}


# ---------- non-streaming (compaction) ----------


async def test_non_stream_response(config: Config):
    response = NS(
        status="completed",
        output_text="SUMMARY",
        output=[NS(type="message"), function_call("c1", "read_file", {"path": "a"})],
        usage=usage(),
    )

    events = await collect(make_llm(config, FakeResponses(response=response)), stream=False)

    assert len(events) == 1
    assert events[0].text_delta.text == "SUMMARY"
    assert events[0].tool_calls == [ToolCall("c1", "read_file", {"path": "a"})]


# ---------- retries ----------


async def test_connection_error_is_retried(config: Config, monkeypatch: pytest.MonkeyPatch):
    async def no_sleep(_):
        pass

    monkeypatch.setattr(llm_base.asyncio, "sleep", no_sleep)
    responses = FakeResponses(
        events=[NS(type="response.completed", response=NS(status="completed", usage=None))],
        errors=[APIConnectionError(request=httpx.Request("POST", "http://x"))],
    )

    events = await collect(make_llm(config, responses))

    assert len(responses.params) == 2
    assert events[-1].type == LLMEventType.Done


# ---------- provider router ----------


async def test_router_uses_provider_from_config(config: Config):
    router = ProviderRouterLLM(config)

    assert isinstance(router.adapter(), OpenAIResponsesLLM)

    config.model.provider = Provider.OPENROUTER
    assert isinstance(router.adapter(), OpenRouterLLM)

    # The same adapter object is reused per provider.
    assert router.adapter() is router.adapter()
    await router.close()


async def test_router_sends_requests_to_current_adapter(config: Config):
    router = ProviderRouterLLM(config)
    responses = FakeResponses(
        events=[
            NS(type="response.output_text.delta", delta="from openai"),
            NS(type="response.completed", response=NS(status="completed", usage=None)),
        ]
    )
    router._adapters[Provider.OPENAI] = make_llm(config, responses)

    events = await collect(router, system_prompt="SYS")

    assert events[0].text_delta.text == "from openai"
    assert responses.params[0]["instructions"] == "SYS"
