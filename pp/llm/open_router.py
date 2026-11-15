import json
from collections.abc import AsyncIterator, Sequence
from copy import deepcopy
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
    ToolCallDelta,
    ToolResultEntry,
    UserMessage,
)
from pp.llm.base import BaseLLM


class OpenRouterLLM(BaseLLM):
    """Adapter for OpenRouter (and other OpenAI-compatible servers) through the Chat Completions API."""

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

        kwargs: dict[str, Any] = {
            "model": self.cfg.model_name,
            "messages": to_openai_messages(entries, system_prompt),
        }
        if self.cfg.temperature is not None:
            kwargs["temperature"] = self.cfg.temperature

        if tools:
            kwargs["tools"] = self._build_tools(tools)
            kwargs["tool_choice"] = "auto"

        async def attempt() -> AsyncIterator[LLMEvent]:
            if stream:
                async for event in self._generate_stream(client, kwargs):
                    yield event
            else:
                yield await self._generate_non_stream(client, kwargs)

        async for event in self._with_retries(attempt):
            yield event

    def _build_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"type": "function", "function": deepcopy(tool)} for tool in tools]

    async def _generate_stream(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> AsyncGenerator[LLMEvent, None]:
        params = {
            "model": kwargs["model"],
            "messages": kwargs["messages"],
            "stream": True,
        }
        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]
        if "tools" in kwargs:
            params["tools"] = kwargs["tools"]
            params["tool_choice"] = kwargs.get("tool_choice", "auto")

        response = await client.chat.completions.create(**params)

        finish_reason: str | None = None
        usage: TokenUsage | None = None
        tool_calls: dict[int, dict[str, Any]] = {}
        tool_calls_started: set[int] = set()

        async for chunk in response:
            # usage only available in the last chunk
            if hasattr(chunk, "usage") and chunk.usage:
                usage = TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                    cached_tokens=(
                        chunk.usage.prompt_tokens_details.cached_tokens or 0 if chunk.usage.prompt_tokens_details else 0
                    ),
                )

            if not chunk.choices:
                continue

            choice = chunk.choices[0]

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            if choice.delta.content:
                yield LLMEvent.stream_text_delta(text_delta=TextDelta(text=choice.delta.content))

            if choice.delta.tool_calls:
                for tool_call_delta in choice.delta.tool_calls:
                    idx = tool_call_delta.index

                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": tool_call_delta.id or "",
                            "name": "",
                            "args": "",
                        }

                    tc = tool_calls[idx]
                    if tool_call_delta.id:
                        tc["id"] = tool_call_delta.id

                    if tool_call_delta.function:
                        if tool_call_delta.function.name:
                            tc["name"] += tool_call_delta.function.name

                        if tc["name"] and idx not in tool_calls_started:
                            yield LLMEvent(
                                type=LLMEventType.ToolCallStart,
                                tool_call_delta=ToolCallDelta(
                                    call_id=tc["id"],
                                    name=tc["name"],
                                ),
                            )
                            tool_calls_started.add(idx)

                        if tool_call_delta.function.arguments:
                            tc["args"] += tool_call_delta.function.arguments
                            yield LLMEvent(
                                type=LLMEventType.ToolCallDelta,
                                tool_call_delta=ToolCallDelta(
                                    call_id=tc["id"],
                                    name=tc["name"],
                                    args_delta=tool_call_delta.function.arguments,
                                ),
                            )

        for _, tc in tool_calls.items():
            yield LLMEvent(
                type=LLMEventType.ToolCallDone,
                tool_call=ToolCall(call_id=tc["id"], name=tc["name"], args=self._parse_tool_call_args(tc["args"])),
            )

        yield LLMEvent.stream_done(finish_reason=finish_reason, usage=usage)

    async def _generate_non_stream(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> LLMEvent:
        params = {
            "model": kwargs["model"],
            "messages": kwargs["messages"],
            "stream": False,
        }
        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]
        if "tools" in kwargs:
            params["tools"] = kwargs["tools"]
            params["tool_choice"] = kwargs.get("tool_choice", "auto")

        response = await client.chat.completions.create(**params)

        choice = response.choices[0]
        message = choice.message

        text_delta = None
        if message.content:
            text_delta = TextDelta(text=message.content)

        tool_calls: list[ToolCall] = []
        for tc in message.tool_calls or []:
            tool_calls.append(
                ToolCall(
                    call_id=tc.id,
                    name=tc.function.name,  # type: ignore[attr-defined]
                    args=self._parse_tool_call_args(tc.function.arguments),  # type: ignore[attr-defined]
                )
            )

        usage: TokenUsage | None = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=(
                    response.usage.prompt_tokens_details.cached_tokens or 0 if response.usage.prompt_tokens_details else 0
                ),
            )

        return LLMEvent.stream_done(finish_reason=choice.finish_reason, usage=usage, text_delta=text_delta, tool_calls=tool_calls)


def to_openai_messages(entries: Sequence[Entry], system_prompt: str | None = None) -> list[dict[str, Any]]:
    """Convert conversation history entries to the OpenAI chat-message format."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for entry in entries:
        if isinstance(entry, UserMessage):
            messages.append({"role": "user", "content": entry.text})
        elif isinstance(entry, AssistantTurn):
            message: dict[str, Any] = {"role": "assistant", "content": entry.text or None}
            if entry.tool_calls:
                message["tool_calls"] = [
                    {"id": call.call_id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.args)}}
                    for call in entry.tool_calls
                ]
            messages.append(message)
        elif isinstance(entry, ToolResultEntry):
            messages.append({"role": "tool", "tool_call_id": entry.call_id, "content": entry.content})
        elif isinstance(entry, Summary):
            messages.append({"role": "system", "content": f"Summary of previous conversation:\n{entry.text}"})
    return messages
