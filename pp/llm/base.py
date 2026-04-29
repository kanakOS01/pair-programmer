import json
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, AsyncGenerator

from pp.config import Config
from pp.domain import LLMEvent


class BaseLLM(ABC):
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._retries = cfg.retries

    @abstractmethod
    def get_client(self) -> Any: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool = True,
    ) -> AsyncGenerator[LLMEvent, None]: ...

    def _parse_tool_call_args(self, args_str: str) -> dict[str, Any]:
        if not args_str:
            return {}

        try:
            return json.loads(args_str)
        except json.JSONDecodeError:
            return {"raw_args": args_str}

    def _build_tools(self, tools: list[dict[str, Any]]) -> list:
        return [{"type": "function", "function": deepcopy(tool)} for tool in tools]
