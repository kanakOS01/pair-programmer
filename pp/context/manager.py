import json
from typing import Any

from pp.config import Config
from pp.domain import Message
from pp.prompts.system import get_system_prompt
from pp.tools.base import Tool
from pp.utils.text import count_tokens


class ContextManager:
    def __init__(self, config: Config, user_memory: str | None, tools: list[Tool] | None) -> None:
        self.config = config
        self.user_memory = user_memory
        self.tools = tools
        self._messages: list[Message] = []
        self._set_system_prompt(get_system_prompt(config, user_memory, tools))

    @property
    def _model_name(self) -> str:
        return self.config.model_name

    def _set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt
        self._system_prompt_tokens = count_tokens(prompt or "", self._model_name)

    def update_system_prompt(self, tools: list[Tool] | None = None) -> None:
        if tools is not None:
            self.tools = tools
        self._set_system_prompt(get_system_prompt(self.config, self.user_memory, self.tools))

    def total_tokens(self) -> int:
        return self._system_prompt_tokens + sum(m.token_count or 0 for m in self._messages)

    def needs_compaction(self) -> bool:
        return self.total_tokens() > self.config.model.context_window * self.config.compaction_threshold

    def add_user_message(self, content: str) -> None:
        message = Message(role="user", content=content, token_count=count_tokens(content, self._model_name))
        self._messages.append(message)

    def add_assistant_message(self, content: str, tool_calls: list[dict[str, Any]] | None = None) -> None:
        message = Message(
            role="assistant",
            content=content or "",
            tool_calls=tool_calls or [],
            token_count=count_tokens((content or "") + (json.dumps(tool_calls) if tool_calls else ""), self._model_name),
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
