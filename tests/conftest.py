from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, AsyncGenerator

import pytest

import pp.utils.text
from pp.config import Config
from pp.domain import Entry, LLMEvent, LLMEventType, TextDelta, TokenUsage, ToolCall, ToolInvocation


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep sessions and memory out of the real home directory.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("API_KEY", "test-key")
    # tiktoken downloads its vocabulary on first use. Use the character estimate, so tests need no network.
    monkeypatch.setattr(pp.utils.text, "get_tokenizer", lambda model: None)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    path = tmp_path / "project"
    path.mkdir()
    return path.resolve()


@pytest.fixture
def config(project: Path) -> Config:
    return Config(cwd=project)


@pytest.fixture
def invocation(project: Path):
    def make(params: dict[str, Any]) -> ToolInvocation:
        return ToolInvocation(cwd=project, params=params)

    return make


def text_turn(text: str) -> list[LLMEvent]:
    return [
        LLMEvent.stream_text_delta(text_delta=TextDelta(text=text)),
        LLMEvent.stream_done(finish_reason="stop", usage=TokenUsage()),
    ]


def tool_turn(text: str, call_id: str, name: str, args: dict[str, Any]) -> list[LLMEvent]:
    events = [LLMEvent.stream_text_delta(text_delta=TextDelta(text=text))] if text else []
    events.append(LLMEvent(type=LLMEventType.ToolCallDone, tool_call=ToolCall(call_id=call_id, name=name, args=args)))
    events.append(LLMEvent.stream_done(finish_reason="tool_calls", usage=TokenUsage(prompt_tokens=1, total_tokens=1)))
    return events


class FakeLLM:
    """Plays back one scripted list of events per agent turn. Compaction calls (tools=None) get a fixed summary."""

    def __init__(self, turns: list[list[LLMEvent]] | None = None, summary: str = "SUMMARY") -> None:
        self.turns = list(turns or [])
        self.summary = summary
        self.calls: list[Sequence[Entry]] = []

    async def generate(
        self,
        entries: Sequence[Entry],
        tools: list[dict[str, Any]] | None,
        system_prompt: str | None = None,
        stream: bool = True,
    ) -> AsyncGenerator[LLMEvent, None]:
        self.calls.append(entries)
        if tools is None:
            yield LLMEvent.stream_done(finish_reason="stop", usage=TokenUsage(), text_delta=TextDelta(text=self.summary))
            return
        for event in self.turns.pop(0):
            yield event

    async def close(self) -> None:
        pass
