from __future__ import annotations

import json
from typing import AsyncGenerator

from pp.agents import BaseAgent
from pp.config import Config
from pp.context.manager import ContextManager
from pp.domain import AgentEvent, AgentEventType, LLMConfig, LLMEventType, TokenUsage, ToolCall
from pp.domain.message import ToolResultMessage
from pp.llm import OpenRouterLLM
from pp.tools.registry import create_default_registry


class CodingAgent(BaseAgent):
    def __init__(self, config: Config) -> None:
        self.config = config
        self.cwd = config.cwd
        self.llm = OpenRouterLLM(
            cfg=LLMConfig(
                model=config.model_name,
                base_url=config.base_url,
                api_key=config.api_key,
                retries=2,
            ),
        )
        self.context_manager = ContextManager()
        self.tool_registry = create_default_registry()

    async def __aenter__(self) -> CodingAgent:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.llm:
            await self.llm.close()

    async def run(self, prompt: str) -> AsyncGenerator[AgentEvent]:
        yield AgentEvent.agent_start(message=prompt)
        self.context_manager.add_user_message(prompt)

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
        response_text = ""

        tool_schemas = self.tool_registry.get_schemas()

        tool_calls: list[ToolCall] = []
        usage: TokenUsage | None = None

        async for event in self.llm.generate(
            messages=self.context_manager.get_messages(),
            tools=tool_schemas,
            stream=True,
        ):
            if event.type == LLMEventType.TextDelta and event.text_delta:
                content = event.text_delta.text
                response_text += content
                yield AgentEvent.agent_text_delta(content=content)

            elif event.type == LLMEventType.ToolCallDone:
                if event.tool_call:
                    tool_calls.append(event.tool_call)

            elif event.type == LLMEventType.Error:
                yield AgentEvent.agent_error(error=event.error or "Unknown error")

            elif event.type == LLMEventType.Done:
                if event.tool_calls:
                    tool_calls.extend(event.tool_calls)
                usage = event.usage

        self.context_manager.add_assistant_message(
            response_text or "",
            tool_calls=[
                {"id": tc.call_id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.args)}}
                for tc in tool_calls
            ]
            if tool_calls
            else None,
        )
        if response_text:
            yield AgentEvent.agent_text_complete(content=response_text)

        tool_call_results: list[ToolResultMessage] = []
        for tool_call in tool_calls:
            yield AgentEvent.tool_call_start(tool_call.call_id, tool_call.name, tool_call.args)

            result = await self.tool_registry.invoke(
                tool_call.name,
                tool_call.args,
                self.cwd,
            )

            yield AgentEvent.tool_call_excecuted(tool_call.call_id, tool_call.name, result)

            tool_call_results.append(
                ToolResultMessage(
                    call_id=tool_call.call_id,
                    content=result.to_model_output(),
                    is_error=not result.ok,
                )
            )

        for tr in tool_call_results:
            self.context_manager.add_tool_result_message(tr.call_id, tr.content)
