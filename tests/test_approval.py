from pathlib import Path

import pytest

from pp.config import Config
from pp.config.config import ApprovalPolicy
from pp.safety.approval import ApprovalDecision, ApprovalManager
from pp.tools.builtin import ReadFileTool, ShellTool, WriteFileTool


def decide(config: Config, tool, params: dict) -> ApprovalDecision:
    manager = ApprovalManager(config)
    return manager.checker.check(manager.create_context(tool, params))


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf build",
        "cd x && rm -rf .",
        "bash -c 'rm -rf .'",
        "/bin/rm x",
        "ls; sudo reboot",
        "echo hi | xargs kill",
        "$(rm x)",
    ],
)
def test_dangerous_command_needs_confirmation_in_auto(project: Path, command: str):
    config = Config(cwd=project, approval="auto")
    assert decide(config, ShellTool(config), {"command": command}) == ApprovalDecision.NEEDS_CONFIRMATION


@pytest.mark.parametrize("command", ["ls -la", "git status", "pytest -q", "echo firmware"])
def test_safe_command_is_approved_in_auto(project: Path, command: str):
    config = Config(cwd=project, approval="auto")
    assert decide(config, ShellTool(config), {"command": command}) == ApprovalDecision.APPROVED


def test_default_policy_is_auto_edit(config: Config):
    assert config.approval.policy == ApprovalPolicy.AUTO_EDIT


def test_auto_edit_approves_writes_and_confirms_shell(config: Config):
    assert decide(config, WriteFileTool(config), {"path": "a.txt", "content": ""}) == ApprovalDecision.APPROVED
    assert decide(config, ShellTool(config), {"command": "ls"}) == ApprovalDecision.NEEDS_CONFIRMATION


def test_on_request_confirms_mutating_tools_only(project: Path):
    config = Config(cwd=project, approval="on-request")
    assert decide(config, WriteFileTool(config), {"path": "a.txt", "content": ""}) == ApprovalDecision.NEEDS_CONFIRMATION
    assert decide(config, ReadFileTool(config), {"path": "a.txt"}) == ApprovalDecision.APPROVED


def test_never_and_yolo(project: Path):
    never = Config(cwd=project, approval="never")
    yolo = Config(cwd=project, approval="yolo")
    assert decide(never, ReadFileTool(never), {"path": "a"}) == ApprovalDecision.REJECTED
    assert decide(yolo, ShellTool(yolo), {"command": "rm -rf x"}) == ApprovalDecision.APPROVED


def test_relative_safe_paths_resolve_against_config_cwd(project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The process runs in a different directory than config.cwd.
    monkeypatch.chdir(tmp_path)
    config = Config(cwd=project, approval={"policy": "on-request", "safe_paths": ["src"]})
    tool = WriteFileTool(config)

    assert decide(config, tool, {"path": "src/a.py", "content": ""}) == ApprovalDecision.APPROVED
    assert decide(config, tool, {"path": "other/a.py", "content": ""}) == ApprovalDecision.NEEDS_CONFIRMATION
    assert decide(config, tool, {"path": "src/../other/a.py", "content": ""}) == ApprovalDecision.NEEDS_CONFIRMATION


async def test_request_confirmation_uses_callback(config: Config):
    manager = ApprovalManager(config, confirmation_callback=lambda c: False)
    from pp.domain import ToolConfirmation

    assert await manager.request_confirmation(ToolConfirmation("shell", {}, "x")) is False
