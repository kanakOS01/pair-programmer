from typing import Any

from pp.config import Config
from pp.domain import Message
from pp.prompts.system import get_system_prompt
from pp.utils.text import count_tokens


class ContextManager:
    def __init__(self, config: Config) -> None:
        self._system_prompt = get_system_prompt(config)
        self._model_name = "stepfun/step-3.5-flash:free"
        self._messages: list[Message] = []

    def add_user_message(self, content: str) -> None:
        message = Message(role="user", content=content, token_count=count_tokens(content, self._model_name))
        self._messages.append(message)

    def add_assistant_message(self, content: str, tool_calls: list[dict[str, Any]] | None = None) -> None:
        message = Message(
            role="assistant",
            content=content or "",
            tool_calls=tool_calls or [],
            token_count=count_tokens(content or "", self._model_name),
        )
        self._messages.append(message)

    def add_tool_result_message(self, tool_call_id: str, content: str) -> None:
        message = Message(
            role="tool", content=content, tool_call_id=tool_call_id, token_count=count_tokens(content, self._model_name)
        )
        self._messages.append(message)

    def get_messages(self) -> list[dict[str, Any]]:
        messages = []

        if self._system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": self._system_prompt,
                }
            )

        for message in self._messages:
            messages.append(message.to_dict())

        return messages
