import asyncio

from pp.llm import OpenRouterLLM
from pp.domain.llm import LLMConfig

async def main():
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
    asyncio.run(main())
