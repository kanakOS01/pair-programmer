import json
from pathlib import Path

import pytest

from pp.config import Config
from pp.context import ConversationHistory
from pp.core.session import Session, get_sessions
from pp.domain import AssistantTurn, Summary, ToolCall, ToolResultEntry, UserMessage
from pp.llm.open_router import to_openai_messages
from pp.utils.paths import get_data_dir


def count(text: str) -> int:
    return len(text) // 4


def call(call_id: str, name: str = "x", **args) -> ToolCall:
    return ToolCall(call_id=call_id, name=name, args=args)


def kinds(history: ConversationHistory) -> list[str]:
    return [type(entry).__name__ for entry in history.entries]


class Summarizer:
    """A summarize function that records what it got."""

    def __init__(self, result: str = "SUMMARY") -> None:
        self.result = result
        self.inputs: list[str] = []

    async def __call__(self, text: str) -> str:
        self.inputs.append(text)
        return self.result


@pytest.fixture
def history() -> ConversationHistory:
    return ConversationHistory(count)


# ---------- tool call / tool result pairing ----------


def test_tool_result_needs_an_open_tool_call(history: ConversationHistory):
    history.add_user("hi")
    with pytest.raises(ValueError, match="No open tool call"):
        history.add_tool_result("c1", "result")


def test_tool_result_cannot_be_added_twice(history: ConversationHistory):
    history.add_assistant_turn("", [call("c1")])
    history.add_tool_result("c1", "ok")
    with pytest.raises(ValueError):
        history.add_tool_result("c1", "again")


def test_open_tool_calls_are_interrupted_by_next_user_message(history: ConversationHistory):
    history.add_user("go")
    history.add_assistant_turn("", [call("c1"), call("c2")])
    history.add_tool_result("c1", "done")

    history.add_user("stop, do something else")

    assert history.entries[-2] == ToolResultEntry("c2", ConversationHistory.INTERRUPTED_RESULT)
    assert history.entries[-1] == UserMessage("stop, do something else")


def test_open_tool_calls_are_interrupted_by_next_assistant_turn(history: ConversationHistory):
    history.add_assistant_turn("", [call("c1")])
    history.add_assistant_turn("next")

    assert kinds(history) == ["AssistantTurn", "ToolResultEntry", "AssistantTurn"]


def test_entries_are_read_only(history: ConversationHistory):
    history.add_user("hi")
    entries = history.entries
    with pytest.raises((AttributeError, TypeError)):
        entries.append(UserMessage("x"))  # type: ignore[attr-defined]


# ---------- tokens ----------


def test_token_count_includes_tool_call_arguments(history: ConversationHistory):
    history.add_assistant_turn("", [call("c1", path="a" * 4000)])
    assert history.token_count() >= 1000


def test_token_count_uses_injected_counter():
    history = ConversationHistory(lambda text: 7)
    history.add_user("a")
    history.add_assistant_turn("b")
    assert history.token_count() == 14


# ---------- compaction ----------


async def test_compaction_does_not_split_tool_call_from_result(history: ConversationHistory):
    history.add_user("u1")
    history.add_assistant_turn("", [call("t1")])
    history.add_tool_result("t1", "r1")
    history.add_assistant_turn("", [call("t2")])
    history.add_tool_result("t2", "r2")

    # keep_last_n=1 would keep only the tool result for t2.
    summary = await history.compact(Summarizer(), keep_last_n=1)

    assert summary == "SUMMARY"
    assert kinds(history) == ["Summary", "AssistantTurn", "ToolResultEntry"]


async def test_compaction_no_op_when_only_cut_point_is_start(history: ConversationHistory):
    history.add_assistant_turn("", [call("t1")])
    history.add_tool_result("t1", "r1")
    summarize = Summarizer()

    assert await history.compact(summarize, keep_last_n=1) is None
    assert summarize.inputs == []
    assert kinds(history) == ["AssistantTurn", "ToolResultEntry"]


async def test_compaction_sends_old_entries_to_summarizer(history: ConversationHistory):
    history.add_user("first question")
    history.add_assistant_turn("first answer", [call("t1", name="read_file", path="a.py")])
    history.add_tool_result("t1", "file body")
    history.add_user("second question")
    summarize = Summarizer()

    await history.compact(summarize, keep_last_n=1)

    text = summarize.inputs[0]
    assert "USER: first question" in text
    assert "read_file" in text and "a.py" in text
    assert "file body" in text
    assert "second question" not in text


async def test_compaction_keeps_entries_added_during_summary(history: ConversationHistory):
    history.add_user("u1")
    history.add_assistant_turn("a1")
    history.add_user("u2")

    async def summarize(text: str) -> str:
        history.add_assistant_turn("added while summarizing")
        return "S"

    await history.compact(summarize, keep_last_n=1)

    assert history.entries[0] == Summary("S")
    assert history.entries[-1] == AssistantTurn("added while summarizing")


async def test_compaction_updates_token_count(history: ConversationHistory):
    for _ in range(10):
        history.add_user("x" * 400)
    before = history.token_count()

    await history.compact(Summarizer("short"), keep_last_n=2)

    assert history.token_count() < before / 3


# ---------- serialization ----------


def test_round_trip(history: ConversationHistory):
    history.add_user("go")
    history.add_assistant_turn("reading", [call("c1", name="read_file", path="a.py")])
    history.add_tool_result("c1", "body")
    history.add_assistant_turn("done")

    restored = ConversationHistory.from_dict(json.loads(json.dumps(history.to_dict())), count)

    assert restored.entries == history.entries
    assert restored.token_count() == history.token_count()


def test_round_trip_keeps_open_tool_calls_open(history: ConversationHistory):
    history.add_assistant_turn("", [call("c1")])

    restored = ConversationHistory.from_dict(history.to_dict(), count)
    restored.add_tool_result("c1", "late result")

    assert restored.entries[-1] == ToolResultEntry("c1", "late result")


OLD_FORMAT = [
    {"role": "system", "content": "Summary of previous conversation:\nearlier work", "tool_calls": []},
    # A tool result without its tool call, as the old compaction bug could leave.
    {"role": "tool", "content": "orphan", "tool_call_id": "gone"},
    {"role": "user", "content": "go", "tool_call_id": None, "tool_calls": []},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}],
    },
    {"role": "tool", "content": "body", "tool_call_id": "c1"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "shell", "arguments": "{bad json"}}],
    },
    # c2 never got a result (the old CLI bug). The next user message closes it.
    {"role": "user", "content": "again", "tool_calls": []},
]


def test_old_format_is_converted():
    history = ConversationHistory.from_dict(OLD_FORMAT, count)

    assert history.entries == (
        Summary("earlier work"),
        UserMessage("go"),
        AssistantTurn("", (call("c1", name="read_file", path="a.py"),)),
        ToolResultEntry("c1", "body"),
        AssistantTurn("", (ToolCall(call_id="c2", name="shell", args={"raw_args": "{bad json"}),)),
        ToolResultEntry("c2", ConversationHistory.INTERRUPTED_RESULT),
        UserMessage("again"),
    )


def test_old_session_file_loads_and_saves_new_format(config: Config):
    sessions = get_data_dir() / "sessions"
    sessions.mkdir(parents=True)
    old = {
        "session_id": "old-1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "messages": OLD_FORMAT,
    }
    (sessions / "old-1.json").write_text(json.dumps(old))

    session = Session(config)
    session.load("old-1")
    session.save()

    saved = json.loads((sessions / "old-1.json").read_text())
    assert "messages" not in saved
    assert saved["history"]["version"] == 2
    assert len(session.history.entries) == 7


def test_session_list_shows_last_text_for_both_formats(config: Config):
    sessions = get_data_dir() / "sessions"
    sessions.mkdir(parents=True)
    old = {"session_id": "old", "updated_at": "2026-01-01T00:00:00+00:00", "messages": OLD_FORMAT}
    (sessions / "old.json").write_text(json.dumps(old))

    session = Session(config)
    session.history.add_user("new format question")
    session.save()

    previews = {s["session_id"]: s["preview"] for s in get_sessions()}
    assert previews["old"] == "again"
    assert previews[session.session_id] == "new format question"


def test_last_text_skips_entries_without_text(history: ConversationHistory):
    history.add_user("question")
    history.add_assistant_turn("", [call("c1")])
    history.add_tool_result("c1", "result")
    assert history.last_text() == "question"


# ---------- OpenAI conversion ----------


def test_to_openai_messages(history: ConversationHistory):
    history.add_user("go")
    history.add_assistant_turn("", [call("c1", name="read_file", path="a.py")])
    history.add_tool_result("c1", "body")
    history.add_assistant_turn("done")

    messages = to_openai_messages(history.entries, system_prompt="SYS")

    assert messages == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "body"},
        {"role": "assistant", "content": "done"},
    ]


def test_summary_becomes_system_message():
    messages = to_openai_messages([Summary("earlier")])
    assert messages == [{"role": "system", "content": "Summary of previous conversation:\nearlier"}]


# ---------- session ----------


async def test_session_compact_counts_usage_and_saves(config: Config, tmp_path: Path):
    from conftest import FakeLLM

    session = Session(config)
    session.llm = FakeLLM(summary="compact summary")  # type: ignore[assignment]
    for i in range(6):
        session.history.add_user(f"message {i}")

    result = await session.compact(keep_last_n=2)

    assert result is not None
    assert result[0] == "compact summary"
    saved = json.loads((get_data_dir() / "sessions" / f"{session.session_id}.json").read_text())
    assert saved["history"]["entries"][0] == {"type": "summary", "text": "compact summary"}


def test_system_prompt_follows_config_changes(config: Config):
    session = Session(config)
    config.developer_instructions = "ALWAYS USE TABS"
    assert "ALWAYS USE TABS" in session.system_prompt()
