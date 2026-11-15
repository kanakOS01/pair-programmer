import asyncio
import os
import signal
import subprocess
import sys
import time
import uuid

import pytest
from conftest import FakeLLM, text_turn
from rich.console import Console

from pp.agents import CodingAgent
from pp.config import Config
from pp.interfaces.cli.main import CLI, coro
from pp.interfaces.cli.themes import DEFAULT_THEME
from pp.tools.builtin import ShellTool

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX signals")


@pytest.fixture(autouse=True)
def restore_sigint():
    yield
    signal.signal(signal.SIGINT, signal.default_int_handler)


def test_ctrl_c_at_prompt_does_not_cancel_later_awaits():
    """The reported bug: Ctrl-C at the prompt, then the next message made pp exit."""

    @coro
    async def main() -> str:
        try:
            # Like input() at the prompt: blocking code that Ctrl-C interrupts.
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.5)
            return "Ctrl-C did not raise KeyboardInterrupt"
        except KeyboardInterrupt:
            pass
        # Like the next message: the first await must not raise CancelledError.
        await asyncio.sleep(0.01)
        return "still running"

    assert main() == "still running"


@pytest.fixture
async def cli(config: Config):
    cli = CLI(config)
    cli.tui.console = Console(record=True, width=100, theme=DEFAULT_THEME)
    async with CodingAgent(config) as agent:
        cli.coding_agent = agent
        yield cli


async def test_ctrl_c_during_turn_stops_only_the_turn(cli: CLI):
    class SlowLLM(FakeLLM):
        async def generate(self, *args, **kwargs):
            await asyncio.sleep(10)
            yield  # pragma: no cover

    cli.coding_agent.session.llm = SlowLLM()
    asyncio.get_running_loop().call_later(0.2, os.kill, os.getpid(), signal.SIGINT)

    start = time.monotonic()
    result = await cli._run_turn("long task")

    assert result is None
    assert time.monotonic() - start < 2
    assert "Interrupted" in cli.tui.console.export_text()
    # The default handler is back, so Ctrl-C at the prompt raises KeyboardInterrupt again.
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler

    # The next turn works.
    cli.coding_agent.session.llm = FakeLLM([text_turn("hello again")])
    assert await cli._run_turn("next") == "hello again"


async def test_cancelled_shell_command_is_stopped(config: Config, invocation):
    marker = f"{50 + uuid.uuid4().int % 1000 / 1000:.3f}"
    task = asyncio.create_task(ShellTool(config).execute(invocation({"command": f"sleep {marker}", "timeout": 60})))
    await asyncio.sleep(0.3)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.2)

    left = subprocess.run(["pgrep", "-f", f"sleep {marker}"], capture_output=True, text=True).stdout.strip()
    assert left == ""
