import asyncio
import sys
from functools import wraps
from typing import Literal

import typer

from pp.agents import CodingAgent
from pp.domain import AgentEventType
from pp.interfaces.cli.tui import TUI, get_console

app = typer.Typer()

console = get_console()


class CLI:
    def __init__(self) -> None:
        self.coding_agent: CodingAgent | None = None
        self.tui: TUI = TUI(console)

    async def run(self, run_type: Literal["one_time", "interactive"] = "interactive", prompt: str = ""):
        if run_type == "one_time":
            return await self._run_once(prompt)
        elif run_type == "interactive":
            return await self._run_interactive()

    async def _run_once(self, prompt: str):
        async with CodingAgent() as agent:
            self.coding_agent = agent
            return await self._process_message(prompt)

    async def _run_interactive(self):
        async with CodingAgent() as agent:
            self.coding_agent = agent

            while True:
                try:
                    inp = console.input("\n[user]>[/user] ").strip()
                    if not inp:
                        continue

                    await self._process_message(inp)

                except KeyboardInterrupt:
                    console.print("\n[dim]Exiting...[/dim]\n")
                    break

                except EOFError:
                    break

        console.print("\n[dim]Goodbye![/dim]\n")

    async def _process_message(self, message: str) -> str | None:
        if not self.coding_agent:
            return None

        assistant_streaming = False

        async for event in self.coding_agent.run(message):
            if event.type == AgentEventType.TextDelta:
                if not assistant_streaming:
                    self.tui.stream_start()
                    assistant_streaming = True

                content = event.data.get("content", "")
                self.tui.stream_delta(content)

            elif event.type == AgentEventType.TextComplete:
                final_response = event.data.get("content", "")
                if assistant_streaming:
                    self.tui.stream_end()
                    assistant_streaming = False

                return final_response

            elif event.type == AgentEventType.Error:
                error = event.data.get("error", "Unkown error")
                self.tui.error(error)

            elif event.type == AgentEventType.ToolCallStart:
                tool_name = event.data.get("name", "Unknown")
                tool = self.coding_agent.tool_registry.get(tool_name)
                tool_type = tool.type.value if tool else None

                self.tui.tool_call_start(
                    call_id=event.data.get("call_id", ""), name=tool_name, tool_type=tool_type, args=event.data.get("args", {})
                )

            elif event.type == AgentEventType.ToolCallDone:
                tool_name = event.data.get("name", "Unknown")
                tool = self.coding_agent.tool_registry.get(tool_name)
                tool_type = tool.type.value if tool else None

                self.tui.tool_call_done(
                    call_id=event.data.get("call_id", ""),
                    name=tool_name,
                    tool_type=tool_type,
                    success=event.data.get("success", False),
                    output=event.data.get("output", ""),
                    error=event.data.get("error", None),
                    meta=event.data.get("metadata", None),
                    truncated=event.data.get("truncated", False),
                )


def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


@app.command()
@coro
async def main(prompt: str = ""):
    """
    Entrypoint of pp (pair-programmer) CLI
    """
    cli = CLI()
    if prompt:
        res = await cli.run(run_type="one_time", prompt=prompt)
        if res is None:
            sys.exit(1)
    else:
        await cli.run()


if __name__ == "__main__":
    app()
