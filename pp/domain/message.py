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
