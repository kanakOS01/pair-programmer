import asyncio
import time
from pathlib import Path

import pytest
from conftest import FakeLLM, text_turn, tool_turn
from rich.console import Console

from pp.agents import CodingAgent
from pp.config import Config
from pp.domain import LLMEventType, ToolConfirmation, UserMessage
from pp.interfaces.cli.main import CLI
from pp.interfaces.cli.themes import DEFAULT_THEME
from pp.llm import OpenAIResponsesLLM, OpenRouterLLM

# ---------- timeout ----------


@pytest.fixture
async def stalled_server():
    """A TCP server that accepts connections and never answers, like an API that hangs."""
    connections = []

    async def handle(reader, writer):
        connections.append(writer)
        await asyncio.sleep(3600)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}/v1"
    for writer in connections:
        writer.close()
    server.close()


@pytest.mark.parametrize("adapter", [OpenAIResponsesLLM, OpenRouterLLM])
async def test_stalled_api_times_out_without_retry(adapter, stalled_server: str, project: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_BASE_URL", stalled_server)
    monkeypatch.setenv("OPENROUTER_BASE_URL", stalled_server)
    provider = "openai" if adapter is OpenAIResponsesLLM else "openrouter"
    config = Config(cwd=project, model={"provider": provider, "timeout": 0.5}, retries=3)
    llm = adapter(config)

    start = time.monotonic()
    events = [e async for e in llm.generate([UserMessage("hi")], None)]
    elapsed = time.monotonic() - start
    await llm.close()

    # One timeout, no retries: 3 attempts would take at least 1.5 seconds plus backoff.
    assert elapsed < 1.4
    assert [e.type for e in events] == [LLMEventType.Error]
    assert "no data for 0.5 seconds" in events[0].error
    assert "model.timeout" in events[0].error


def test_client_has_timeout_and_no_sdk_retries(project: Path):
    config = Config(cwd=project, model={"timeout": 42})
    client = OpenAIResponsesLLM(config).get_client()

    assert client.max_retries == 0
    assert client.timeout.read == 42
    assert client.timeout.connect == 15


# ---------- thinking spinner ----------


@pytest.fixture
async def cli(config: Config):
    cli = CLI(config)
    cli.tui.console = Console(record=True, width=100, theme=DEFAULT_THEME, force_terminal=True)
    async with CodingAgent(config) as agent:
        cli.coding_agent = agent
        agent.session.tool_executor.confirm = cli._confirm_tool_call
        yield cli


async def test_thinking_spinner_shows_while_waiting(cli: CLI):
    seen_live = []

    class SlowLLM(FakeLLM):
        async def generate(self, *args, **kwargs):
            # While the model works, the spinner must be running.
            seen_live.append(cli.tui._live_thinking is not None)
            await asyncio.sleep(0.3)
            async for event in super().generate(*args, **kwargs):
                yield event

    cli.coding_agent.session.llm = SlowLLM([tool_turn("", "c1", "list_dir", {}), text_turn("done")])

    result = await cli._process_message("go")

    assert result == "done"
    # Once for the first request, once after the tool result.
    assert seen_live == [True, True]
    # Nothing is left running afterwards.
    assert cli.tui._live_thinking is None
    assert "Thinking" in cli.tui.console.export_text()


async def test_spinner_stops_on_error(cli: CLI):
    class BrokenLLM(FakeLLM):
        async def generate(self, *args, **kwargs):
            raise RuntimeError("boom")
            yield  # pragma: no cover

    cli.coding_agent.session.llm = BrokenLLM()

    with pytest.raises(RuntimeError):
        await cli._process_message("go")

    assert cli.tui._live_thinking is None


async def test_live_displays_stop_before_approval_question(cli: CLI, config: Config, monkeypatch: pytest.MonkeyPatch):
    config.approval.policy = "on-request"  # type: ignore[assignment]
    state_at_prompt = {}

    def fake_input(prompt: str = "") -> str:
        tui = cli.tui
        state_at_prompt["live"] = [tui._live_thinking, tui._live_tool_call, tui._live_stream]
        return "y"

    monkeypatch.setattr(cli.tui.console, "input", fake_input)
    cli.coding_agent.session.llm = FakeLLM(
        [tool_turn("", "c1", "write_file", {"path": "a.txt", "content": "x"}), text_turn("written")]
    )

    result = await cli._process_message("write a file")

    assert result == "written"
    assert state_at_prompt["live"] == [None, None, None]


def test_confirm_question_is_visible_with_live_tool_panel(cli: CLI, monkeypatch: pytest.MonkeyPatch):
    cli.tui.tool_call_start(call_id="c1", name="shell", tool_type="shell", args={"command": "ls"})
    cli.tui.thinking_start()
    monkeypatch.setattr(cli.tui.console, "input", lambda prompt="": "n")

    approved = cli._confirm_tool_call(ToolConfirmation("shell", {"command": "ls"}, "Execute shell"))

    assert approved is False
    assert cli.tui._live_tool_call is None
    assert "Requires Confirmation" in cli.tui.console.export_text()
