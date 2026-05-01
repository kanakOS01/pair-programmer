from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ValidationError
from pydantic.json_schema import model_json_schema

from pp.config.config import Config
from pp.domain import ToolConfirmation, ToolInvocation, ToolResult, ToolType


class Tool(ABC):
    name: str
    description: str
    type: ToolType

    def __init__(self, config: Config) -> None:
        self.config = config

    @property
    @abstractmethod
    def schema(self) -> dict[str, Any] | type[BaseModel]: ...

    @abstractmethod
    async def execute(self, invocation: ToolInvocation) -> ToolResult: ...

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        schema = self.schema

        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            return []

        try:
            schema.model_validate(params)
            return []
        except ValidationError as ve:
            return [f"Parameter '{'.'.join(str(x) for x in e['loc'])}': {e['msg']}" for e in ve.errors()]
        except Exception as e:
            return [str(e)]

    def is_mutating(self) -> bool:
        return self.type in (
            ToolType.WRITE,
            ToolType.MEMORY,
            ToolType.SHELL,
            ToolType.NETWORK,
        )

    async def get_confirmation(self, invocation: ToolInvocation) -> ToolConfirmation | None:
        if not self.is_mutating():
            return None

        return ToolConfirmation(
            tool_name=self.name,
            params=invocation.params,
            description=f"Execute {self.name}",
        )

    def to_openai_schema(self) -> dict[str, Any]:
        schema = self.schema

        if isinstance(schema, dict):
            json_schema: dict[str, Any] = {
                "name": self.name,
                "description": self.description,
            }
            json_schema["parameters"] = schema["parameters"] if "parameters" in schema else schema
            return json_schema

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema = model_json_schema(schema, mode="serialization")
            return {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": json_schema.get("properties", {}),
                    "required": json_schema.get("required", []),
                },
            }

        raise ValueError(f"Invalid schema type for tool {self.name}: {type(schema)}")
