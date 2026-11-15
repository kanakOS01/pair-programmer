from pathlib import Path
from typing import Any

import pytest

from pp.config import Config
from pp.config.config import ApprovalPolicy, HookConfig
from pp.domain import ToolCall, ToolConfirmation, ToolEffects, ToolInvocation, ToolResult, ToolType
from pp.hooks import HookSystem
from pp.tools.base import Tool
from pp.tools.executor import ToolExecutor
from pp.tools.registry import create_default_registry


class Confirmer:
    """A confirm function that records what it was asked and gives a fixed answer."""

    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.asked: list[ToolConfirmation] = []

    def __call__(self, confirmation: ToolConfirmation) -> bool:
        self.asked.append(confirmation)
        return self.answer


def make_executor(config: Config, confirm=None) -> ToolExecutor:
    return ToolExecutor(create_default_registry(config), config, HookSystem(config), confirm)


def shell(command: str) -> ToolCall:
    return ToolCall(call_id="c", name="shell", args={"command": command})


def write(path: str) -> ToolCall:
    return ToolCall(call_id="c", name="write_file", args={"path": path, "content": "x"})


def read(path: str) -> ToolCall:
    return ToolCall(call_id="c", name="read_file", args={"path": path})


# ---------- confirmation ----------


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf build",
        "cd x && rm -rf .",
        "bash -c 'rm -rf .'",
        "/bin/rm x",
        "ls; sudo reboot-later",
        "echo hi | xargs kill",
        "$(rm x)",
    ],
)
async def test_dangerous_command_asks_in_auto(project: Path, command: str):
    config = Config(cwd=project, approval="auto")
    confirm = Confirmer(False)

    result = await make_executor(config, confirm).run(shell(command))

    assert len(confirm.asked) == 1
    assert result.metadata["approval_decision"] == "rejected_by_user"


@pytest.mark.parametrize("command", ["echo ok", "git --version", "echo firmware"])
async def test_safe_command_runs_without_asking_in_auto(project: Path, command: str):
    config = Config(cwd=project, approval="auto")
    confirm = Confirmer(False)

    result = await make_executor(config, confirm).run(shell(command))

    assert confirm.asked == []
    assert result.ok


async def test_confirmation_shows_tool_and_params(project: Path):
    config = Config(cwd=project, approval="on-request")
    confirm = Confirmer(True)

    result = await make_executor(config, confirm).run(write("a.txt"))

    assert result.ok
    assert confirm.asked[0].tool_name == "write_file"
    assert confirm.asked[0].params["path"] == "a.txt"


async def test_async_confirm_function_is_awaited(project: Path):
    config = Config(cwd=project, approval="on-request")

    async def confirm(confirmation: ToolConfirmation) -> bool:
        return False

    result = await make_executor(config, confirm).run(write("a.txt"))

    assert not result.ok
    assert not (project / "a.txt").exists()


async def test_no_confirm_function_rejects(project: Path):
    config = Config(cwd=project, approval="on-request")

    result = await make_executor(config, confirm=None).run(write("a.txt"))

    assert not result.ok
    assert result.metadata["approval_decision"] == "no_confirmation"
    assert not (project / "a.txt").exists()


# ---------- policies ----------


def test_default_policy_is_auto_edit(config: Config):
    assert config.approval.policy == ApprovalPolicy.AUTO_EDIT


async def test_auto_edit_writes_without_asking_and_asks_for_shell(config: Config, project: Path):
    confirm = Confirmer(True)
    executor = make_executor(config, confirm)

    assert (await executor.run(write("a.txt"))).ok
    assert confirm.asked == []

    assert (await executor.run(shell("echo hi"))).ok
    assert [c.tool_name for c in confirm.asked] == ["shell"]


async def test_on_request_asks_for_writes_not_reads(project: Path):
    config = Config(cwd=project, approval="on-request")
    (project / "a.txt").write_text("x")
    confirm = Confirmer(True)
    executor = make_executor(config, confirm)

    await executor.run(read("a.txt"))
    assert confirm.asked == []
    await executor.run(write("a.txt"))
    assert len(confirm.asked) == 1


async def test_never_rejects_and_yolo_runs(project: Path):
    never = Config(cwd=project, approval="never")
    yolo = Config(cwd=project, approval="yolo")

    rejected = await make_executor(never).run(read("a.txt"))
    assert rejected.metadata["approval_decision"] == "rejected"

    ran = await make_executor(yolo).run(shell("echo yolo"))
    assert ran.ok and "yolo" in ran.output


# ---------- safe paths ----------


async def test_relative_safe_paths_resolve_against_config_cwd(project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The process runs in a different directory than config.cwd.
    monkeypatch.chdir(tmp_path)
    config = Config(cwd=project, approval={"policy": "on-request", "safe_paths": ["src"]})
    confirm = Confirmer(True)
    executor = make_executor(config, confirm)

    await executor.run(write("src/a.py"))
    assert confirm.asked == []

    await executor.run(write("other/a.py"))
    await executor.run(write("src/../other/b.py"))
    assert len(confirm.asked) == 2


async def test_safe_paths_do_not_approve_shell(project: Path):
    # A shell command can touch any file, so a safe project root must not approve it.
    config = Config(cwd=project, approval={"policy": "on-request", "safe_paths": ["."]})
    confirm = Confirmer(False)

    await make_executor(config, confirm).run(shell("echo hi"))

    assert len(confirm.asked) == 1


# ---------- declared effects ----------


class RecordingTool(Tool):
    """A tool with custom effects, to test that approval reads effects, not parameter names."""

    name = "custom"
    description = "test"
    type = ToolType.WRITE
    schema = {"type": "object", "properties": {"target": {"type": "string"}}}

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        return ToolResult.success_result("ran")

    def effects(self, params: dict[str, Any]) -> ToolEffects:
        return ToolEffects(paths=(Path(params["target"]),))


async def test_approval_uses_declared_effects(project: Path):
    config = Config(cwd=project, approval={"policy": "on-request", "safe_paths": ["src"]})
    executor = make_executor(config, Confirmer(False))
    executor.registry.register(RecordingTool(config))

    inside = await executor.run(ToolCall("c", "custom", {"target": "src/x"}))
    outside = await executor.run(ToolCall("c", "custom", {"target": "elsewhere/x"}))

    assert inside.ok
    assert not outside.ok


async def test_failing_effects_become_error_result(project: Path):
    class BrokenEffects(RecordingTool):
        name = "broken"

        def effects(self, params: dict[str, Any]) -> ToolEffects:
            raise RuntimeError("boom")

    config = Config(cwd=project)
    executor = make_executor(config)
    executor.registry.register(BrokenEffects(config))

    result = await executor.run(ToolCall("c", "broken", {"target": "x"}))

    assert not result.ok
    assert "boom" in result.error


# ---------- lookup, validation, errors ----------


async def test_unknown_tool(config: Config):
    result = await make_executor(config).run(ToolCall("c", "nope", {}))
    assert not result.ok and "not found" in result.error


async def test_invalid_params(config: Config):
    result = await make_executor(config).run(ToolCall("c", "read_file", {}))
    assert not result.ok and "Invalid parameters" in result.error


async def test_tool_crash_becomes_error_result(config: Config):
    class Crashing(RecordingTool):
        name = "crashing"

        async def execute(self, invocation: ToolInvocation) -> ToolResult:
            raise RuntimeError("tool crashed")

    executor = make_executor(config, Confirmer(True))
    executor.registry.register(Crashing(config))

    result = await executor.run(ToolCall("c", "crashing", {"target": "x"}))

    assert not result.ok and "tool crashed" in result.error


# ---------- hooks ----------


def hook_config(project: Path, policy: str = "yolo") -> Config:
    config = Config(cwd=project, approval=policy, hooks_enabled=True)
    config.hooks = [
        HookConfig(name="b", trigger="before tool", script='echo "before $PP_TOOL_NAME" >> hooks.log'),
        HookConfig(name="a", trigger="after tool", script='echo "after $PP_TOOL_NAME $PP_TOOL_SUCCESS" >> hooks.log'),
    ]
    return config


async def test_hooks_run_around_the_tool(project: Path):
    await make_executor(hook_config(project)).run(shell("echo hi"))

    assert (project / "hooks.log").read_text().splitlines() == ["before shell", "after shell True"]


async def test_hooks_do_not_run_for_rejected_calls(project: Path):
    await make_executor(hook_config(project, policy="never")).run(shell("echo hi"))

    assert not (project / "hooks.log").exists()
