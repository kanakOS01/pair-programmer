from __future__ import annotations

import json
from typing import AsyncGenerator

from pp.agents import BaseAgent
from pp.config import Config
from pp.core.session import Session
from pp.domain import AgentEvent, AgentEventType, LLMEventType, TokenUsage, ToolCall
from pp.domain.message import ToolResultMessage


class CodingAgent(BaseAgent):
    def __init__(self, config: Config) -> None:
        self.session: Session = Session(config)

    async def __aenter__(self) -> CodingAgent:
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if hasattr(self.session, "mcp_manager") and self.session.mcp_manager:
            await self.session.mcp_manager.close()
        if self.session.llm:
            await self.session.llm.close()

    async def run(self, prompt: str) -> AsyncGenerator[AgentEvent]:
        yield AgentEvent.agent_start(message=prompt)
        self.session.context_manager.add_user_message(prompt)

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
        for _ in range(self.session.config.max_turns):
            _ = self.session.increment_turn()

            response_text = ""

            tool_schemas = self.session.tool_registry.get_schemas()

            tool_calls: list[ToolCall] = []
            usage: TokenUsage | None = None

            async for event in self.session.llm.generate(
                messages=self.session.context_manager.get_messages(),
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

            self.session.context_manager.add_assistant_message(
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

            # if no tool calls are present we may come out of the loop
            if not tool_calls:
                break

            tool_call_results: list[ToolResultMessage] = []
            for tool_call in tool_calls:
                yield AgentEvent.tool_call_start(tool_call.call_id, tool_call.name, tool_call.args)

                result = await self.session.tool_registry.invoke(
                    tool_call.name,
                    tool_call.args,
                    self.session.config.cwd,
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
                self.session.context_manager.add_tool_result_message(tr.call_id, tr.content)
