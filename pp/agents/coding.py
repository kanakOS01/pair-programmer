from __future__ import annotations

import logging
from typing import AsyncGenerator

from pp.agents import BaseAgent
from pp.config import Config
from pp.core.session import Session
from pp.domain import AgentEvent, AgentEventType, LLMEventType, TokenUsage, ToolCall

logger = logging.getLogger(__name__)


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
        try:
            await self.session.hook_system.run_hooks("before_agent", {"agent_prompt": prompt})
            self.session.history.add_user(prompt)
            self.session.save()

            final_response = None
            async for event in self._loop():
                yield event

                if event.type == AgentEventType.TextComplete:
                    final_response = event.data.get("content")

            await self.session.hook_system.run_hooks(
                "after_agent",
                {
                    "agent_prompt": prompt,
                    "agent_response": final_response or "",
                },
            )
            yield AgentEvent.agent_done(final_response, self.session.token_usage)
        except Exception as e:
            await self.session.hook_system.run_hooks("on_error", {"agent_prompt": prompt, "error": str(e)})
            raise e

    async def _loop(self) -> AsyncGenerator[AgentEvent, None]:
        """
        Multiturn interaction with LLM
        """
        for _ in range(self.session.config.max_turns):
            _ = self.session.increment_turn()

            if self.session.needs_compaction():
                await self._compact()

            response_text = ""

            tool_schemas = self.session.tool_registry.get_schemas()

            tool_calls: list[ToolCall] = []
            usage: TokenUsage | None = None

            async for event in self.session.llm.generate(
                self.session.history.entries,
                tools=tool_schemas,
                system_prompt=self.session.system_prompt(),
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
                    error_msg = event.error or "Unknown error"
                    await self.session.hook_system.run_hooks("on_error", {"error": error_msg})
                    yield AgentEvent.agent_error(error=error_msg)

                elif event.type == LLMEventType.Done:
                    if event.tool_calls:
                        tool_calls.extend(event.tool_calls)
                    usage = event.usage

            # An empty turn (for example after an LLM error) is not recorded.
            if response_text or tool_calls:
                self.session.history.add_assistant_turn(response_text, tool_calls)
            if usage:
                self.session.accumulate_usage(usage)
            self.session.save()
            if response_text:
                yield AgentEvent.agent_text_complete(content=response_text)

            # if no tool calls are present we may come out of the loop
            if not tool_calls:
                break

            for tool_call in tool_calls:
                yield AgentEvent.tool_call_start(tool_call.call_id, tool_call.name, tool_call.args)

                result = await self.session.tool_executor.run(tool_call)

                yield AgentEvent.tool_call_excecuted(tool_call.call_id, tool_call.name, result)

                # Record each result at once. If the run stops later, the finished results are kept.
                self.session.history.add_tool_result(tool_call.call_id, result.to_model_output())
                self.session.save()

    async def _compact(self) -> None:
        try:
            await self.session.compact()
        except Exception:
            # A failed compaction must not stop the run. The next LLM call can still succeed.
            logger.exception("Automatic compaction failed")
