from typing import AsyncGenerator
from abc import ABC, abstractmethod
from typing import Any

from pp.domain.llm import LLMConfig, StreamEvent


class BaseLLM(ABC):
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._retries = cfg.retries or 3

    
    @abstractmethod
    def get_client(self) -> Any:
        ...
    

    @abstractmethod
    async def close(self) -> None:
        ...


    @abstractmethod
    def generate(self, messages: list[dict[str, Any]], stream: bool = True) -> AsyncGenerator[StreamEvent, None]:
        ...
