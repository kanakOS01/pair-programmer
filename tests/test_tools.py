import socket
import subprocess
import sys
import uuid
from pathlib import Path

import httpx
import pytest

import pp.tools.builtin.read_url as read_url_module
from pp.config import Config
from pp.tools.builtin import (
    ApplyPatchTool,
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    ReadUrlTool,
    ShellTool,
    TodoTool,
    WriteFileTool,
)

# ---------- file tools ----------


@pytest.mark.parametrize(
    ("tool_cls", "params"),
    [
        (ReadFileTool, {"path": "../secret.txt"}),
        (WriteFileTool, {"path": "../secret.txt", "content": "x"}),
        (EditFileTool, {"path": "../secret.txt", "old_str": "a", "new_str": "b"}),
        (ApplyPatchTool, {"path": "../secret.txt", "patches": [{"old_str": "a", "new_str": "b"}]}),
        (ListDirTool, {"path": ".."}),
    ],
)
async def test_file_tools_reject_paths_outside_cwd(config: Config, invocation, tool_cls, params, project: Path):
    (project.parent / "secret.txt").write_text("a")

    result = await tool_cls(config).execute(invocation(params))

    assert not result.ok
    assert "outside" in result.error
    assert (project.parent / "secret.txt").read_text() == "a"


async def test_write_read_edit_patch_inside_cwd(config: Config, invocation, project: Path):
    assert (await WriteFileTool(config).execute(invocation({"path": "sub/a.txt", "content": "one\ntwo\n"}))).ok

    read = await ReadFileTool(config).execute(invocation({"path": "sub/a.txt"}))
    assert read.ok and "one" in read.output

    edit = await EditFileTool(config).execute(invocation({"path": "sub/a.txt", "old_str": "one", "new_str": "1"}))
    assert edit.ok

    patch = await ApplyPatchTool(config).execute(
        invocation({"path": "sub/a.txt", "patches": [{"old_str": "two", "new_str": "2"}]})
    )
    assert patch.ok
    assert (project / "sub/a.txt").read_text() == "1\n2\n"


# ---------- todos ----------


async def test_todo_add_list_done_clear(config: Config, invocation):
    todo = TodoTool(config)

    add = await todo.execute(invocation({"action": "add", "message": "write tests", "id": "t1"}))
    assert add.ok

    listed = await todo.execute(invocation({"action": "LIST"}))
    assert listed.ok
    assert "write tests" in listed.output
    assert listed.metadata["todos"][0]["status"] == "pending"

    assert (await todo.execute(invocation({"action": "done", "id": "t1"}))).ok
    listed = await todo.execute(invocation({"action": "list"}))
    assert listed.metadata["todos"][0]["status"] == "done"

    cleared = await todo.execute(invocation({"action": "Clear"}))
    assert cleared.ok
    empty = await todo.execute(invocation({"action": "list"}))
    assert empty.ok and empty.metadata == {"todos": []}


# ---------- shell ----------


@pytest.mark.parametrize(
    "command",
    ["rm -rf /", 'rm  -rf  "/"', "sudo rm -r ~", "rm -rf /*", "dd if=/dev/zero of=x", "mkfs.ext4 /dev/sda", "shutdown -h now"],
)
async def test_shell_blocks_destructive_commands(config: Config, invocation, command: str):
    result = await ShellTool(config).execute(invocation({"command": command}))
    assert not result.ok
    assert "blocked" in result.error


@pytest.mark.parametrize("command", ["rm -rf /tmp/pp-test-nothing-here", "echo rebooting"])
def test_shell_blocklist_allows_safe_commands(config: Config, command: str):
    assert ShellTool(config)._find_blocked_pattern(command) is None


async def test_shell_cwd_outside_project_is_rejected(config: Config, invocation):
    result = await ShellTool(config).execute(invocation({"command": "pwd", "cwd": ".."}))
    assert not result.ok
    assert "outside" in result.error


async def test_shell_truncates_failed_output(config: Config, invocation):
    tool = ShellTool(config)
    tool.MAX_OUTPUT_SIZE_BYTES = 100

    result = await tool.execute(invocation({"command": "yes | head -c 5000; exit 3"}))

    assert not result.ok
    assert result.exit_code == 3
    assert result.truncated
    assert len(result.error) < 500


async def test_shell_hides_secret_env_vars(config: Config, invocation, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MY_API_KEY", "s3cret")
    result = await ShellTool(config).execute(invocation({"command": "env"}))
    assert "s3cret" not in result.output


@pytest.mark.skipif(sys.platform == "win32", reason="uses process groups")
async def test_shell_timeout_kills_child_processes(config: Config, invocation):
    marker = f"{40 + uuid.uuid4().int % 1000 / 1000:.3f}"  # a unique sleep time, to find our processes

    result = await ShellTool(config).execute(invocation({"command": f"sleep {marker} & sleep {marker}; wait", "timeout": 1}))

    assert not result.ok and "timed out" in result.error
    left = subprocess.run(["pgrep", "-f", f"sleep {marker}"], capture_output=True, text=True).stdout.strip()
    assert left == ""


# ---------- read_url ----------


@pytest.fixture
def fake_dns(monkeypatch: pytest.MonkeyPatch):
    """Map host names to IP addresses without real DNS."""
    hosts: dict[str, str] = {}

    def getaddrinfo(host, port, *args, **kwargs):
        ip = hosts.get(host, host)
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, port or 80))]

    monkeypatch.setattr(read_url_module.socket, "getaddrinfo", getaddrinfo)
    return hosts


@pytest.fixture
def fake_http(monkeypatch: pytest.MonkeyPatch):
    """Serve responses from a dict of url -> httpx.Response, without the network."""
    routes: dict[str, httpx.Response] = {}
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return routes[str(request.url)]

    real_client = httpx.AsyncClient

    def client(*args, **kwargs):
        return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(read_url_module.httpx, "AsyncClient", client)
    return routes, requested


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1", "::ffff:127.0.0.1", "0.0.0.0"])
async def test_read_url_blocks_private_addresses(config: Config, invocation, fake_dns, ip: str):
    fake_dns["internal.example"] = ip
    result = await ReadUrlTool(config).execute(invocation({"url": "http://internal.example/"}))
    assert not result.ok
    assert "private" in result.error


async def test_read_url_blocks_redirect_to_private_address(config: Config, invocation, fake_dns, fake_http):
    fake_dns["public.example"] = "93.184.216.34"
    routes, requested = fake_http
    routes["http://public.example/"] = httpx.Response(302, headers={"location": "http://169.254.169.254/latest/"})

    result = await ReadUrlTool(config).execute(invocation({"url": "http://public.example/"}))

    assert not result.ok
    assert "private" in result.error
    assert requested == ["http://public.example/"]


async def test_read_url_follows_public_redirect(config: Config, invocation, fake_dns, fake_http):
    fake_dns["public.example"] = "93.184.216.34"
    routes, _ = fake_http
    routes["http://public.example/"] = httpx.Response(301, headers={"location": "/new"})
    routes["http://public.example/new"] = httpx.Response(200, text="hello")

    result = await ReadUrlTool(config).execute(invocation({"url": "http://public.example/"}))

    assert result.ok
    assert result.output == "hello"


async def test_read_url_stops_redirect_loop(config: Config, invocation, fake_dns, fake_http):
    fake_dns["public.example"] = "93.184.216.34"
    routes, _ = fake_http
    routes["http://public.example/"] = httpx.Response(302, headers={"location": "/"})

    result = await ReadUrlTool(config).execute(invocation({"url": "http://public.example/"}))

    assert not result.ok
    assert "redirects" in result.error


async def test_read_url_rejects_non_http_scheme(config: Config, invocation):
    result = await ReadUrlTool(config).execute(invocation({"url": "file:///etc/passwd"}))
    assert not result.ok
