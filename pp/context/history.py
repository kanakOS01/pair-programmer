from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from pp.domain.llm import ToolCall
from pp.domain.message import AssistantTurn, Entry, Summary, ToolResultEntry, UserMessage

logger = logging.getLogger(__name__)

CountTokens = Callable[[str], int]
Summarize = Callable[[str], Awaitable[str]]

FORMAT_VERSION = 2
OLD_SUMMARY_PREFIX = "Summary of previous conversation:\n"


class ConversationHistory:
    """The ordered record of user messages, assistant turns, tool results and summaries in one session.

    Invariant: every tool call has exactly one tool result after it. A tool call that is still open
    when the next user message arrives gets an "interrupted" result.
    """

    INTERRUPTED_RESULT = "Interrupted by user. The tool call did not finish."

    def __init__(self, count_tokens: CountTokens) -> None:
        self._count_tokens = count_tokens
        self._entries: list[Entry] = []
        self._tokens: list[int] = []
        # Tool calls of the last assistant turn that do not have a result yet, in call order.
        self._open_calls: list[str] = []

    # ======= Adding entries ======= #

    def add_user(self, text: str) -> None:
        self._close_open_calls()
        self._append(UserMessage(text))

    def add_assistant_turn(self, text: str, tool_calls: Sequence[ToolCall] = ()) -> None:
        self._close_open_calls()
        self._append(AssistantTurn(text, tuple(tool_calls)))
        self._open_calls = [call.call_id for call in tool_calls]

    def add_tool_result(self, call_id: str, content: str) -> None:
        if call_id not in self._open_calls:
            raise ValueError(f"No open tool call with id {call_id!r}")
        self._open_calls.remove(call_id)
        self._append(ToolResultEntry(call_id, content))

    # ======= Reading ======= #

    @property
    def entries(self) -> Sequence[Entry]:
        return tuple(self._entries)

    def token_count(self) -> int:
        return sum(self._tokens)

    def last_text(self) -> str:
        """The text of the last user message or assistant turn that has text. Empty if there is none."""
        for entry in reversed(self._entries):
            if isinstance(entry, (UserMessage, AssistantTurn)) and entry.text.strip():
                return entry.text.strip()
        return ""

    # ======= Compaction ======= #

    async def compact(self, summarize: Summarize, keep_last_n: int = 4) -> str | None:
        """Replace the oldest entries with one summary. Keep at least the last keep_last_n entries.

        The cut never separates a tool call from its tool result. Return the summary text,
        or None if there is nothing to compact.
        """
        keep_last_n = max(0, keep_last_n)
        snapshot = list(self._entries)

        # Move the cut back until the kept entries do not start with a tool result.
        cut = len(snapshot) - keep_last_n
        while 0 < cut < len(snapshot) and isinstance(snapshot[cut], ToolResultEntry):
            cut -= 1
        if cut <= 0:
            return None

        summary_text = (await summarize(_format_for_summary(snapshot[:cut]))).strip()

        # Entries added while summarize() ran are after the cut, so they are kept.
        summary = Summary(summary_text)
        self._entries = [summary, *self._entries[cut:]]
        self._tokens = [self._count_entry(summary), *self._tokens[cut:]]
        return summary_text

    # ======= Serialization ======= #

    def to_dict(self) -> dict[str, Any]:
        return {"version": FORMAT_VERSION, "entries": [_entry_to_dict(entry) for entry in self._entries]}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | list[dict[str, Any]], count_tokens: CountTokens) -> ConversationHistory:
        """Build a history from to_dict() output, or from the old session format (a list of OpenAI-style messages)."""
        history = cls(count_tokens)
        if isinstance(data, list):
            entries = _entries_from_old_format(data)
        else:
            entries = [_entry_from_dict(item) for item in data.get("entries", [])]

        for entry in entries:
            history._restore(entry)
        return history

    # ======= Helpers ======= #

    def _append(self, entry: Entry) -> None:
        self._entries.append(entry)
        self._tokens.append(self._count_entry(entry))

    def _close_open_calls(self) -> None:
        for call_id in list(self._open_calls):
            self.add_tool_result(call_id, self.INTERRUPTED_RESULT)

    def _restore(self, entry: Entry) -> None:
        if isinstance(entry, UserMessage):
            self.add_user(entry.text)
        elif isinstance(entry, AssistantTurn):
            self.add_assistant_turn(entry.text, entry.tool_calls)
        elif isinstance(entry, ToolResultEntry):
            try:
                self.add_tool_result(entry.call_id, entry.content)
            except ValueError:
                # Old session files can contain tool results without their tool call. Skip them.
                logger.warning(f"Skipping tool result without a tool call: {entry.call_id}")
        else:
            self._append(entry)

    def _count_entry(self, entry: Entry) -> int:
        if isinstance(entry, AssistantTurn):
            calls = json.dumps([_call_to_dict(call) for call in entry.tool_calls]) if entry.tool_calls else ""
            return self._count_tokens(entry.text + calls)
        if isinstance(entry, ToolResultEntry):
            return self._count_tokens(entry.content)
        return self._count_tokens(entry.text)


def _format_for_summary(entries: Sequence[Entry]) -> str:
    lines = []
    for entry in entries:
        if isinstance(entry, UserMessage):
            lines.append(f"USER: {entry.text}")
        elif isinstance(entry, AssistantTurn):
            text = entry.text
            if entry.tool_calls:
                text += f"\n[Tool Calls: {[_call_to_dict(call) for call in entry.tool_calls]}]"
            lines.append(f"ASSISTANT: {text}")
        elif isinstance(entry, ToolResultEntry):
            lines.append(f"TOOL (ID: {entry.call_id}): {entry.content}")
        else:
            lines.append(f"SUMMARY OF EARLIER CONVERSATION: {entry.text}")
    return "\n\n".join(lines)


def _call_to_dict(call: ToolCall) -> dict[str, Any]:
    return {"id": call.call_id, "name": call.name, "args": call.args}


def _entry_to_dict(entry: Entry) -> dict[str, Any]:
    if isinstance(entry, UserMessage):
        return {"type": "user", "text": entry.text}
    if isinstance(entry, AssistantTurn):
        return {"type": "assistant", "text": entry.text, "tool_calls": [_call_to_dict(call) for call in entry.tool_calls]}
    if isinstance(entry, ToolResultEntry):
        return {"type": "tool_result", "call_id": entry.call_id, "content": entry.content}
    return {"type": "summary", "text": entry.text}


def _entry_from_dict(data: dict[str, Any]) -> Entry:
    kind = data["type"]
    if kind == "user":
        return UserMessage(data["text"])
    if kind == "assistant":
        calls = tuple(ToolCall(call_id=c["id"], name=c["name"], args=c.get("args", {})) for c in data.get("tool_calls", []))
        return AssistantTurn(data["text"], calls)
    if kind == "tool_result":
        return ToolResultEntry(data["call_id"], data["content"])
    if kind == "summary":
        return Summary(data["text"])
    raise ValueError(f"Unknown history entry type: {kind!r}")


def _entries_from_old_format(messages: list[dict[str, Any]]) -> list[Entry]:
    """Convert the old session format: OpenAI-style messages with role, content, tool_calls and tool_call_id."""
    entries: list[Entry] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        if role == "user":
            entries.append(UserMessage(content))
        elif role == "assistant":
            calls = []
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                try:
                    args = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"raw_args": function.get("arguments")}
                calls.append(ToolCall(call_id=call["id"], name=function.get("name", ""), args=args))
            entries.append(AssistantTurn(content, tuple(calls)))
        elif role == "tool":
            entries.append(ToolResultEntry(message.get("tool_call_id") or "", content))
        elif role == "system":
            entries.append(Summary(content.removeprefix(OLD_SUMMARY_PREFIX)))
    return entries
