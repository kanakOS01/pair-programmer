import asyncio
from functools import wraps

import typer

from pp.llm import OpenRouterLLM
from pp.domain.llm import LLMConfig

app = typer.Typer()


class CLI:
    def __init__(self) -> None:
        pass

    
    def run_single(self, prompt: str):
        pass


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
    cfg = LLMConfig(
        
    )
    llm = OpenRouterLLM(cfg)

    messages = [
        {"role": "user", "content": "Hello, how are you?"},
    ]
    async for event in llm.generate(
        messages=messages,
        stream=False,
    ):
        print(event)


if __name__ == "__main__":
    app()
