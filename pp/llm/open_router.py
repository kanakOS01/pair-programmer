from openai import APIError
from openai import APIConnectionError
import asyncio
from openai import RateLimitError
from typing import override, Any, AsyncGenerator

from openai import AsyncOpenAI

from pp.domain.llm import LLMConfig, TextDelta, TokenUsage, EventType, StreamEvent
from pp.llm.base import BaseLLM


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
    async def generate(self, messages: list[dict[str, Any]], stream: bool = True) -> AsyncGenerator[StreamEvent, None]:
        client = self.get_client()
        kwargs = {
            "model": self.cfg.model,
            "messages": messages,
        }

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
                    yield StreamEvent(
                        type=EventType.Error,
                        error=f"Rate limit exceeded. {rle}",
                    )
                    return
                await asyncio.sleep(2 ** attempt)
            
            except APIConnectionError as ace:
                if attempt == self._retries - 1:
                    yield StreamEvent(
                        type=EventType.Error,
                        error=f"API connection error. {ace}",
                    )
                    return
                await asyncio.sleep(2 ** attempt)
            
            except APIError as ae:
                if attempt == self._retries - 1:
                    yield StreamEvent(
                        type=EventType.Error,
                        error=f"Error decoding response. {ae}",
                    )
                    return
                await asyncio.sleep(2 ** attempt)


    async def _generate_stream(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> AsyncGenerator[StreamEvent, None]:
        response = await client.chat.completions.create(
            model=kwargs["model"],
            messages=kwargs["messages"],
            stream=True,
        )

        finish_reason: str | None = None
        usage: TokenUsage | None = None

        async for chunk in response:
            # usage only available in the last chunk
            if hasattr(chunk, "usage") and chunk.usage:
                usage = TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                    cached_tokens=chunk.usage.prompt_tokens_details.cached_tokens,
                )
            
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            
            if choice.finish_reason:
                finish_reason = choice.finish_reason   
        
            if choice.delta.content:
                yield StreamEvent(
                    type=EventType.TextDelta,
                    text_delta=TextDelta(text=choice.delta.content),
                )
        
        yield StreamEvent(
            type=EventType.Done,
            finish_reason=finish_reason,
            usage=usage,
        )


    async def _generate_non_stream(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> StreamEvent:
        response = await client.chat.completions.create(
            model=kwargs["model"],
            messages=kwargs["messages"],
            stream=False,
        )
        
        choice = response.choices[0]
        message = choice.message

        text_delta = None
        if message.content:
            text_delta = TextDelta(text=message.content)

        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            cached_tokens=response.usage.prompt_tokens_details.cached_tokens,
        )

        return StreamEvent(
            type=EventType.Done,
            text_delta=text_delta,
            usage=usage,
            finish_reason=choice.finish_reason
        )
        