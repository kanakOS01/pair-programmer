from collections.abc import Sequence
from typing import Any, AsyncGenerator, override

from pp.config import Config
from pp.config.config import Provider
from pp.domain import Entry, LLMEvent
from pp.llm.base import BaseLLM
from pp.llm.open_router import OpenRouterLLM
from pp.llm.openai_responses import OpenAIResponsesLLM

ADAPTERS: dict[Provider, type[BaseLLM]] = {
    Provider.OPENAI: OpenAIResponsesLLM,
    Provider.OPENROUTER: OpenRouterLLM,
}


class ProviderRouterLLM(BaseLLM):
    """Sends each request to the adapter of the provider in the current config.

    The provider is read on each request, so a /config change takes effect without a restart.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self._adapters: dict[Provider, BaseLLM] = {}

    def adapter(self) -> BaseLLM:
        provider = self.cfg.model.provider
        if provider not in self._adapters:
            self._adapters[provider] = ADAPTERS[provider](self.cfg)
        return self._adapters[provider]

    @override
    async def close(self) -> None:
        for adapter in self._adapters.values():
            await adapter.close()
        self._adapters.clear()

    @override
    async def generate(
        self,
        entries: Sequence[Entry],
        tools: list[dict[str, Any]] | None,
        system_prompt: str | None = None,
        stream: bool = True,
    ) -> AsyncGenerator[LLMEvent, None]:
        async for event in self.adapter().generate(entries, tools, system_prompt, stream):
            yield event


def create_llm(cfg: Config) -> BaseLLM:
    return ProviderRouterLLM(cfg)
