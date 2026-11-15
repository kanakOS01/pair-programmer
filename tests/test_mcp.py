import sys
import textwrap
import time
from pathlib import Path

import pytest

from pp.config import Config
from pp.config.config import MCPServerConfig
from pp.tools.mcp import MCPClient, MCPManager

CRASHING_SERVER = """
import sys
print("Traceback (most recent call last):", file=sys.stderr)
print("PermissionError: [Errno 13] Permission denied: '/Users/macbook'", file=sys.stderr)
sys.exit(1)
"""

# A minimal MCP server: answers initialize, tools/list and tools/call over stdin/stdout.
ECHO_SERVER = """
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    if msg["method"] == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "echo", "version": "0"}}
    elif msg["method"] == "tools/list":
        result = {"tools": [{"name": "echo", "description": "Echo text", "inputSchema": {"type": "object"}}]}
    elif msg["method"] == "tools/call":
        result = {"content": [{"type": "text", "text": msg["params"]["arguments"]["text"]}]}
    print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
"""


def server_script(tmp_path: Path, name: str, code: str) -> str:
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(code))
    return str(path)


async def test_crashing_server_fails_fast_with_stderr(tmp_path: Path):
    client = MCPClient("broken", sys.executable, [server_script(tmp_path, "crash", CRASHING_SERVER)])

    start = time.monotonic()
    with pytest.raises(RuntimeError) as error:
        await client.connect(timeout=15)

    assert time.monotonic() - start < 5
    assert "exit code 1" in str(error.value)
    assert "Permission denied: '/Users/macbook'" in str(error.value)


async def test_manager_registers_and_calls_server_tools(tmp_path: Path, project: Path):
    script = server_script(tmp_path, "echo", ECHO_SERVER)
    config = Config(cwd=project, mcp_servers={"demo": MCPServerConfig(command=sys.executable, args=[script])})
    manager = MCPManager(config)

    try:
        await manager.initialize()
        tools = await manager.get_tools()
        assert [t.name for t in tools] == ["mcp_demo_echo"]

        result = await manager.call_tool("demo", "echo", {"text": "hello"})
        assert result.ok and result.output == "hello"
    finally:
        await manager.close()


async def test_manager_keeps_running_when_one_server_fails(tmp_path: Path, project: Path):
    config = Config(
        cwd=project,
        mcp_servers={
            "broken": MCPServerConfig(command=sys.executable, args=[server_script(tmp_path, "crash", CRASHING_SERVER)]),
            "demo": MCPServerConfig(command=sys.executable, args=[server_script(tmp_path, "echo", ECHO_SERVER)]),
        },
    )
    manager = MCPManager(config)

    try:
        await manager.initialize()
        assert [t.name for t in await manager.get_tools()] == ["mcp_demo_echo"]
    finally:
        await manager.close()


HANGING_SERVER = """
import time
time.sleep(60)
"""


def server(tmp_path: Path, name: str, code: str, **kwargs) -> MCPServerConfig:
    return MCPServerConfig(command=sys.executable, args=[server_script(tmp_path, name, code)], **kwargs)


async def test_slow_server_error_names_startup_timeout(tmp_path: Path, project: Path):
    config = Config(cwd=project, mcp_servers={"slow": server(tmp_path, "hang", HANGING_SERVER, startup_timeout=0.5)})
    manager = MCPManager(config)

    try:
        await manager.initialize()
    finally:
        await manager.close()

    assert not manager.clients["slow"].connected
    assert "0.5 seconds" in manager.errors["slow"]
    assert "startup_timeout in [mcp_servers.slow]" in manager.errors["slow"]


async def test_servers_start_at_the_same_time(tmp_path: Path, project: Path):
    config = Config(
        cwd=project,
        mcp_servers={
            "slow1": server(tmp_path, "hang1", HANGING_SERVER, startup_timeout=1),
            "slow2": server(tmp_path, "hang2", HANGING_SERVER, startup_timeout=1),
            "slow3": server(tmp_path, "hang3", HANGING_SERVER, startup_timeout=1),
        },
    )
    manager = MCPManager(config)

    start = time.monotonic()
    try:
        await manager.initialize()
    finally:
        await manager.close()

    # One timeout, not three in a row.
    assert time.monotonic() - start < 2.5
    assert set(manager.errors) == {"slow1", "slow2", "slow3"}


async def test_errors_are_cleared_on_next_initialize(tmp_path: Path, project: Path):
    config = Config(cwd=project, mcp_servers={"demo": server(tmp_path, "echo", ECHO_SERVER)})
    manager = MCPManager(config)
    manager.errors["demo"] = "old error"

    try:
        await manager.initialize()
    finally:
        await manager.close()

    assert manager.errors == {}


async def test_cli_shows_mcp_failures(tmp_path: Path, project: Path):
    from rich.console import Console

    from pp.interfaces.cli.main import CLI
    from pp.interfaces.cli.tui import DEFAULT_THEME

    config = Config(
        cwd=project,
        mcp_servers={
            "broken": server(tmp_path, "crash", CRASHING_SERVER),
            "demo": server(tmp_path, "echo", ECHO_SERVER),
        },
    )
    cli = CLI(config)
    cli.tui.console = Console(record=True, width=120, theme=DEFAULT_THEME)

    from contextlib import AsyncExitStack

    async with AsyncExitStack() as stack:
        agent = await cli._start_agent(stack)
        summary = cli._mcp_summary_lines()
        cli.tui.show_mcp_errors(agent.session.mcp_manager.errors)

    output = cli.tui.console.export_text()
    assert summary == ["MCP broken: failed (see below)", "MCP demo: connected, 1 tools"]
    assert "MCP server 'broken' did not start" in output
    assert "Permission denied: '/Users/macbook'" in output
