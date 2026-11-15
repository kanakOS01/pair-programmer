from pathlib import Path

import pytest
from pydantic import ValidationError

from pp.config import Config, load_config
from pp.config.config import ApprovalPolicy, HookConfig
from pp.hooks import HookSystem
from pp.utils.errors import ConfigError


def write_project_config(project: Path, text: str) -> None:
    (project / ".pair-programmer").mkdir(exist_ok=True)
    (project / ".pair-programmer" / "config.toml").write_text(text)


def test_invalid_project_toml_is_skipped(project: Path):
    write_project_config(project, "not = = toml")

    config = load_config(project)

    assert config.cwd == project


def test_project_config_is_loaded(project: Path):
    write_project_config(project, '[model]\nname = "x/y"\n[approval]\npolicy = "on-request"\n')

    config = load_config(project)

    assert config.model_name == "x/y"
    assert config.approval.policy == ApprovalPolicy.ON_REQUEST


def test_approval_accepts_plain_string(project: Path):
    write_project_config(project, 'approval = "yolo"\n')
    assert load_config(project).approval.policy == ApprovalPolicy.YOLO


def test_config_error_without_details():
    error = ConfigError("bad", config_file="f.toml")
    assert error.details == {"config_file": "f.toml"}
    assert str(error).startswith("bad")


def test_retries_must_be_positive():
    with pytest.raises(ValidationError):
        Config(retries=0)


async def test_hook_env_values_are_truncated(config: Config, project: Path):
    config.hooks_enabled = True
    config.hooks = [HookConfig(name="h", trigger="after tool", script='printf %s "$PP_TOOL_RESULT" > out.txt')]
    hooks = HookSystem(config)

    await hooks.run_hooks("after_tool", {"tool_result": "x" * 1_000_000})

    out = (project / "out.txt").read_text()
    assert len(out) < HookSystem.MAX_ENV_VALUE_CHARS + 100
    assert out.endswith("[truncated]")
