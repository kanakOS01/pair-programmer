import asyncio
from typing import Any, AsyncGenerator, override

from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError

from pp.domain import LLMConfig, LLMEvent, LLMEventType, TextDelta, TokenUsage, ToolCall, ToolCallDelta
from pp.llm import BaseLLM


class OpenRouterLLM(BaseLLM):
    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__(cfg)
        self._client: AsyncOpenAI | None = None

    @override
    def get_client(self) -> Any:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.cfg.base_url,
                api_key=self.cfg.api_key,
            )
        return self._client

    @override
    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    @override
    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        stream: bool = True,
    ) -> AsyncGenerator[LLMEvent, None]:
        client = self.get_client()

        kwargs = {
            "model": self.cfg.model,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = self._build_tools(tools)
            kwargs["tool_choice"] = "auto"

        for attempt in range(self._retries):
            try:
                if stream:
                    async for event in self._generate_stream(client, kwargs):
                        yield event
                else:
                    yield await self._generate_non_stream(client, kwargs)

                return

            except RateLimitError as rle:
                if attempt == self._retries - 1:
                    yield LLMEvent.stream_error(error=f"Rate limit exceeded. {rle}")
                    return
                await asyncio.sleep(2**attempt)

            except APIConnectionError as ace:
                if attempt == self._retries - 1:
                    yield LLMEvent.stream_error(error=f"API connection error. {ace}")
                    return
                await asyncio.sleep(2**attempt)

            except APIError as ae:
                if attempt == self._retries - 1:
                    yield LLMEvent.stream_error(error=f"Error decoding response. {ae}")
                    return
                await asyncio.sleep(2**attempt)

    async def _generate_stream(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> AsyncGenerator[LLMEvent, None]:
        response = await client.chat.completions.create(
            model=kwargs["model"],
            messages=kwargs["messages"],
            tools=kwargs["tools"],
            stream=True,
        )

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
        response = await client.chat.completions.create(
            model=kwargs["model"],
            messages=kwargs["messages"],
            tools=kwargs["tools"],
            stream=False,
        )

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
