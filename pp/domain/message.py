from dataclasses import dataclass
from typing import Any


@dataclass
class Message:
    role: str
    content: str
    token_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
        }
