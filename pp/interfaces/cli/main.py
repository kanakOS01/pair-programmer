from pp.domain import AgentEventType
from pp.agents import CodingAgent
import asyncio
from functools import wraps

import typer

app = typer.Typer()


class CLI:
    def __init__(self) -> None:
        self.coding_agent: CodingAgent | None = None


    
    async def run_once(self, prompt: str):
        async with CodingAgent() as agent:
            self.coding_agent = agent

            await self._process_message(prompt)            


    async def _process_message(self, message: str) -> str | None:
        if not self.coding_agent:
            return None
        
        async for event in self.coding_agent.run(message):
            if event.type == AgentEventType.TextDelta:
                content = event.data.get("content", "")



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
        await cli.run_once(prompt)


if __name__ == "__main__":
    app()
