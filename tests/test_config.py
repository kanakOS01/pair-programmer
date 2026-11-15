import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from pp.config import Config, load_config
from pp.config.config import ApprovalPolicy, HookConfig, Provider
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

    removed = ("API_KEY", "BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL")
    environ = {k: v for k, v in os.environ.items() if k not in removed}
    monkeypatch.setattr(os, "environ", environ)
    return environ


def write_env(directory: Path, text: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".env").write_text(text)


def test_api_key_from_project_env_file(project: Path, clean_environ):
    write_env(project / ".pair-programmer", "OPENAI_API_KEY=project-key\n")

    config = load_config(project)

    assert config.api_key == "project-key"
    assert config.get_validation_errors() == []


def test_api_key_from_home_env_file(project: Path, clean_environ):
    write_env(Path.home() / ".pair-programmer", "OPENAI_API_KEY=home-key\n")

    assert load_config(project).api_key == "home-key"


def test_project_env_file_wins_over_home(project: Path, clean_environ):
    write_env(Path.home() / ".pair-programmer", "OPENAI_API_KEY=home-key\nOPENAI_BASE_URL=https://home.example\n")
    write_env(project / ".pair-programmer", "OPENAI_API_KEY=project-key\n")

    config = load_config(project)

    assert config.api_key == "project-key"
    assert config.base_url == "https://home.example"


def test_exported_variable_wins_over_env_file(project: Path, clean_environ):
    clean_environ["OPENAI_API_KEY"] = "exported-key"
    write_env(project / ".pair-programmer", "OPENAI_API_KEY=file-key\n")

    assert load_config(project).api_key == "exported-key"


def test_env_file_syntax(project: Path, clean_environ):
    write_env(
        project / ".pair-programmer",
        "# comment\n\nexport OPENAI_API_KEY=\"quoted key\"\nOPENAI_BASE_URL = 'https://x.example'\nnot a valid line\n",
    )

    config = load_config(project)

    assert config.api_key == "quoted key"
    assert config.base_url == "https://x.example"


def test_missing_key_error_names_the_env_file(project: Path, clean_environ):
    errors = load_config(project).get_validation_errors()
    assert any(".pair-programmer/.env" in e for e in errors)


# ---------- providers ----------


def test_default_model_is_openai_sol(project: Path):
    config = Config(cwd=project)
    assert config.model.provider == Provider.OPENAI
    assert config.model_name == "gpt-6.1-sol"
    assert config.temperature is None
    assert config.base_url == "https://api.openai.com/v1"


def test_openai_ignores_openrouter_variables(project: Path, clean_environ):
    clean_environ.update(API_KEY="sk-or-old", BASE_URL="https://openrouter.ai/api/v1")
    config = Config(cwd=project)

    assert config.api_key == ""
    assert config.base_url == "https://api.openai.com/v1"
    assert any("OPENAI_API_KEY" in e for e in config.get_validation_errors())


def test_openrouter_keys_and_old_api_key_name(project: Path, clean_environ):
    config = Config(cwd=project, model={"provider": "openrouter", "name": "openai/gpt-oss-120b"})

    clean_environ["API_KEY"] = "old-name-key"
    assert config.api_key == "old-name-key"
    assert config.base_url == "https://openrouter.ai/api/v1"

    clean_environ["OPENROUTER_API_KEY"] = "new-name-key"
    assert config.api_key == "new-name-key"


def test_model_settings_are_validated_on_assignment(project: Path):
    config = Config(cwd=project)

    config.model.provider = "openrouter"  # type: ignore[assignment]
    assert config.model.provider == Provider.OPENROUTER

    config.model.temp = "0.5"  # type: ignore[assignment]
    assert config.model.temp == 0.5

    with pytest.raises(ValidationError):
        config.model.provider = "anthropic"  # type: ignore[assignment]
    with pytest.raises(ValidationError):
        config.model.reasoning_effort = "extreme"  # type: ignore[assignment]


@pytest.mark.skipif(sys.platform == "win32", reason="uses process groups")
async def test_hooks_run_in_own_process_group(config: Config, project: Path):
    import os

    config.hooks_enabled = True
    config.hooks = [HookConfig(name="pgid", trigger="before agent", script="ps -o pgid= -p $$ > pgid.txt")]

    await HookSystem(config).run_hooks("before_agent", {})

    assert int((project / "pgid.txt").read_text()) != os.getpgid(0)
