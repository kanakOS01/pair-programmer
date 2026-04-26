from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str
    content: str
    token_count: int | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {"role": self.role}

        if self.tool_call_id:
            res["tool_call_id"] = self.tool_call_id

        if self.tool_calls:
            res["tool_calls"] = self.tool_calls

        if self.content:
            res["content"] = self.content

        return res


@dataclass
class ToolResultMessage:
    call_id: str
    content: str
    is_error: bool = False

    def to_openai_message(self) -> dict[str, Any]:
        return {
            "role": "tool",
            "call_id": self.call_id,
            "content": self.content,
        }
