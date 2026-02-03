from typing import AsyncGenerator
from typing import Any

from pp.agents.base import BaseAgent
from pp.llm.base import BaseLLM
from pp.domain.agent import AgentEvent


class CodingAgent(BaseAgent):
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    
    def run(self, prompt: str) -> Any:
        pass


    async def _loop(self) -> AsyncGenerator[AgentEvent, None]:
        pass