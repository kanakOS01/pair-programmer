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
        if stream:
            async for event in self._generate_stream(client, kwargs):
                yield event
        else:
            yield await self._generate_non_stream(client, kwargs)
    

    async def _generate_stream(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> AsyncGenerator[StreamEvent, None]:
        response = await client.chat.completions.create(
            model=kwargs["model"],
            messages=kwargs["messages"],
            stream=True,
        )

        async for chunk in response:
            yield chunk



    
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
        