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


# ---------- .env files ----------


@pytest.fixture
def clean_environ(monkeypatch: pytest.MonkeyPatch):
    """Give the test its own copy of os.environ without API_KEY, so loaded values cannot leak to other tests."""
    import os

    environ = {k: v for k, v in os.environ.items() if k not in ("API_KEY", "BASE_URL")}
    monkeypatch.setattr(os, "environ", environ)
    return environ


def write_env(directory: Path, text: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".env").write_text(text)


def test_api_key_from_project_env_file(project: Path, clean_environ):
    write_env(project / ".pair-programmer", "API_KEY=project-key\n")

    config = load_config(project)

    assert config.api_key == "project-key"
    assert config.get_validation_errors() == []


def test_api_key_from_home_env_file(project: Path, clean_environ):
    write_env(Path.home() / ".pair-programmer", "API_KEY=home-key\n")

    assert load_config(project).api_key == "home-key"


def test_project_env_file_wins_over_home(project: Path, clean_environ):
    write_env(Path.home() / ".pair-programmer", "API_KEY=home-key\nBASE_URL=https://home.example\n")
    write_env(project / ".pair-programmer", "API_KEY=project-key\n")

    config = load_config(project)

    assert config.api_key == "project-key"
    assert config.base_url == "https://home.example"


def test_exported_variable_wins_over_env_file(project: Path, clean_environ):
    clean_environ["API_KEY"] = "exported-key"
    write_env(project / ".pair-programmer", "API_KEY=file-key\n")

    assert load_config(project).api_key == "exported-key"


def test_env_file_syntax(project: Path, clean_environ):
    write_env(
        project / ".pair-programmer",
        "# comment\n\nexport API_KEY=\"quoted key\"\nBASE_URL = 'https://x.example'\nnot a valid line\n",
    )

    config = load_config(project)

    assert config.api_key == "quoted key"
    assert config.base_url == "https://x.example"


def test_missing_key_error_names_the_env_file(project: Path, clean_environ):
    errors = load_config(project).get_validation_errors()
    assert any(".pair-programmer/.env" in e for e in errors)
