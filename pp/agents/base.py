from abc import ABC, abstractmethod
from typing import AsyncGenerator

from pp.domain import AgentEvent


class BaseAgent(ABC):
    @abstractmethod
    def run(self, prompt: str) ->  AsyncGenerator[AgentEvent]:
        ...