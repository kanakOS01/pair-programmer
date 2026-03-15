import sys

from pp.domain import AgentEventType
from pp.agents import CodingAgent
import asyncio
from functools import wraps

import typer

from pp.interfaces.cli.tui import TUI, get_console

app = typer.Typer()

console = get_console()


class CLI:
    def __init__(self) -> None:
        self.coding_agent: CodingAgent | None = None
        self.tui: TUI = TUI(console)

    
    async def run_once(self, prompt: str):
        async with CodingAgent() as agent:
            self.coding_agent = agent
            return await self._process_message(prompt)            


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
    messages = [{"role": "user", "content": prompt}]
    if prompt:
        res = await cli.run_once(prompt)
        if res is None:
            sys.exit(1)


if __name__ == "__main__":
    app()
