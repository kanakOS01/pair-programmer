import json
from collections.abc import AsyncIterator, Sequence
from typing import Any, AsyncGenerator, override

from openai import AsyncOpenAI

from pp.config import Config
from pp.domain import (
    AssistantTurn,
    Entry,
    LLMEvent,
    LLMEventType,
    Summary,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolResultEntry,
    UserMessage,
)
from pp.llm.base import BaseLLM


class OpenAIResponsesLLM(BaseLLM):
    """Adapter for the OpenAI API through the Responses API. OpenAI requires the Responses API for tool calls."""

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self._client: AsyncOpenAI | None = None

    def get_client(self) -> Any:
        if self._client is None:
            self._client = self._new_client()
        return self._client

    @override
    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    @override
    async def generate(
        self,
        entries: Sequence[Entry],
        tools: list[dict[str, Any]] | None,
        system_prompt: str | None = None,
        stream: bool = True,
    ) -> AsyncGenerator[LLMEvent, None]:
        client = self.get_client()
        params = self._build_params(entries, tools, system_prompt)

        async def attempt() -> AsyncIterator[LLMEvent]:
            if stream:
                async for event in self._generate_stream(client, params):
                    yield event
            else:
                yield await self._generate_non_stream(client, params)

        async for event in self._with_retries(attempt):
            yield event

    def _build_params(
        self, entries: Sequence[Entry], tools: list[dict[str, Any]] | None, system_prompt: str | None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.cfg.model_name,
            "input": to_responses_input(entries),
            # The conversation history is sent on each request, so OpenAI does not need to store responses.
            "store": False,
        }
        if system_prompt:
            params["instructions"] = system_prompt
        if tools:
            params["tools"] = to_responses_tools(tools)
        if self.cfg.temperature is not None:
            params["temperature"] = self.cfg.temperature
        if self.cfg.model.reasoning_effort is not None:
            params["reasoning"] = {"effort": self.cfg.model.reasoning_effort}
        return params

    async def _generate_stream(self, client: AsyncOpenAI, params: dict[str, Any]) -> AsyncGenerator[LLMEvent, None]:
        response_stream = await client.responses.create(**params, stream=True)

        async for event in response_stream:
            if event.type == "response.output_text.delta":
                yield LLMEvent.stream_text_delta(text_delta=TextDelta(text=event.delta))

            elif event.type == "response.output_item.done" and event.item.type == "function_call":
                yield LLMEvent(type=LLMEventType.ToolCallDone, tool_call=self._tool_call(event.item))

            elif event.type == "response.completed":
                yield LLMEvent.stream_done(finish_reason=event.response.status, usage=_usage(event.response.usage))

            elif event.type == "response.incomplete":
                details = event.response.incomplete_details
                reason = details.reason if details else "unknown"
                yield LLMEvent.stream_done(finish_reason=f"incomplete: {reason}", usage=_usage(event.response.usage))

            elif event.type == "response.failed":
                error = event.response.error
                yield LLMEvent.stream_error(error=f"Response failed. {error.message if error else 'Unknown error'}")

            elif event.type == "error":
                yield LLMEvent.stream_error(error=f"API error. {event.message}")

    async def _generate_non_stream(self, client: AsyncOpenAI, params: dict[str, Any]) -> LLMEvent:
        response = await client.responses.create(**params)

        tool_calls = [self._tool_call(item) for item in response.output if item.type == "function_call"]
        text = response.output_text
        return LLMEvent.stream_done(
            finish_reason=response.status,
            usage=_usage(response.usage),
            text_delta=TextDelta(text=text) if text else None,
            tool_calls=tool_calls,
        )

    def _tool_call(self, item: Any) -> ToolCall:
        return ToolCall(call_id=item.call_id, name=item.name, args=self._parse_tool_call_args(item.arguments))


def to_responses_input(entries: Sequence[Entry]) -> list[dict[str, Any]]:
    """Convert conversation history entries to Responses API input items."""
    items: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, UserMessage):
            items.append({"role": "user", "content": entry.text})
        elif isinstance(entry, AssistantTurn):
            if entry.text:
                items.append({"role": "assistant", "content": entry.text})
            for call in entry.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": json.dumps(call.args),
                    }
                )
        elif isinstance(entry, ToolResultEntry):
            items.append({"type": "function_call_output", "call_id": entry.call_id, "output": entry.content})
        elif isinstance(entry, Summary):
            items.append({"role": "developer", "content": f"Summary of previous conversation:\n{entry.text}"})
    return items


def to_responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert tool schemas ({name, description, parameters}) to Responses API function tools."""
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            # Strict mode needs every property required and no extra properties. Our schemas have optional
            # parameters, and MCP schemas are not under our control, so strict mode is off.
            "strict": False,
        }
        for tool in tools
    ]


def _usage(usage: Any) -> TokenUsage | None:
    if usage is None:
        return None
    details = usage.input_tokens_details
    return TokenUsage(
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cached_tokens=(details.cached_tokens or 0) if details else 0,
    )
