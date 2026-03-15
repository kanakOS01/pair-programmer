from __future__ import annotations

from typing import AsyncGenerator

from pp.agents import BaseAgent
from pp.domain import AgentEvent, AgentEventType, LLMConfig, LLMEventType
from pp.llm import OpenRouterLLM


class CodingAgent(BaseAgent):
    def __init__(self) -> None:
        self.llm = OpenRouterLLM(
            cfg=LLMConfig(
                model="stepfun/step-3.5-flash:free",
                base_url="https://openrouter.ai/api/v1",
                retries=2,
            ),
        )

    async def __aenter__(self) -> CodingAgent:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.llm:
            await self.llm.close()

    async def run(self, prompt: str) -> AsyncGenerator[AgentEvent]:
        yield AgentEvent.agent_start(message=prompt)

        final_response = None
        async for event in self._loop():
            yield event

            if event.type == AgentEventType.TextComplete:
                final_response = event.data.get("content")

        yield AgentEvent.agent_done(final_response)

    async def _loop(self) -> AsyncGenerator[AgentEvent, None]:
        """
        Multiturn interaction with LLM
        """
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
        ]

        response_text = ""

        async for event in self.llm.generate(messages=messages, stream=True):
            if event.type == LLMEventType.TextDelta and event.text_delta:
                content = event.text_delta.text
                response_text += content
                yield AgentEvent.agent_text_delta(content=content)
            elif event.type == LLMEventType.Error:
                yield AgentEvent.agent_error(error=event.error or "Unknown error")

        if response_text:
            yield AgentEvent.agent_text_complete(content=response_text)
