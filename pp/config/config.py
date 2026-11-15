from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator


class Provider(str, Enum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"


# Environment variables for each provider. The first variable that is set wins.
API_KEY_VARS: dict[Provider, tuple[str, ...]] = {
    Provider.OPENAI: ("OPENAI_API_KEY",),
    # API_KEY is the old name for the OpenRouter key. It still works.
    Provider.OPENROUTER: ("OPENROUTER_API_KEY", "API_KEY"),
}
BASE_URL_VARS: dict[Provider, tuple[str, ...]] = {
    Provider.OPENAI: ("OPENAI_BASE_URL",),
    Provider.OPENROUTER: ("OPENROUTER_BASE_URL", "BASE_URL"),
}
DEFAULT_BASE_URLS: dict[Provider, str] = {
    Provider.OPENAI: "https://api.openai.com/v1",
    Provider.OPENROUTER: "https://openrouter.ai/api/v1",
}


class ModelConfig(BaseModel):
    # Validate on assignment, so that /config and /model changes are checked and converted too.
    model_config = ConfigDict(validate_assignment=True)

    provider: Provider = Provider.OPENAI
    name: str = "gpt-6.1-sol"
    # None: do not send a temperature. Reasoning models can reject the parameter.
    temp: float | None = Field(default=None, ge=0.0, le=2.0)
    # OpenAI only. None: use the model's default effort.
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] | None = None
    context_window: int = Field(256_000)
    # Seconds to wait for data from the API: for the connection, and between parts of a streamed answer.
    # Reasoning models can think for a long time before they send anything, so keep it generous.
    timeout: float = Field(default=300.0, gt=0)


class ShellEnvironmentPolicy(BaseModel):
    ignore_default_excludes: bool = False  # tell whether to ignore default excludes like secret with *KEY*
    exclude_patterns: list[str] = Field(default_factory=lambda: ["*KEY*", "*TOKEN*", "*SECRET*"])
    set_vars: dict[str, str] = Field(default_factory=dict)


class ApprovalPolicy(str, Enum):
    ON_REQUEST = "on-request"
    ON_FAILURE = "on-failure"
    AUTO = "auto"
    AUTO_EDIT = "auto-edit"
    NEVER = "never"
    YOLO = "yolo"


class ApprovalConfig(BaseModel):
    policy: ApprovalPolicy = ApprovalPolicy.AUTO_EDIT
    safe_tools: list[str] = Field(default_factory=list)
    requires_approval_tools: list[str] = Field(default_factory=list)
    requires_approval_types: list[str] = Field(default_factory=list)
    safe_paths: list[str] = Field(default_factory=list)
    dangerous_commands: list[str] = Field(default_factory=list)


def _coerce_approval(v: Any) -> ApprovalConfig:
    if isinstance(v, ApprovalConfig):
        return v
    if isinstance(v, dict):
        return ApprovalConfig(**v)
    if isinstance(v, (str, ApprovalPolicy)):
        return ApprovalConfig(policy=ApprovalPolicy(v))
    raise ValueError(f"Cannot coerce {v} to ApprovalConfig")


class MCPServerConfig(BaseModel):
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    # Seconds to wait for the server to answer its first request. Increase it if the server
    # downloads packages when it starts (for example with uvx).
    startup_timeout: float = Field(default=15.0, gt=0)


class HookConfig(BaseModel):
    name: str
    trigger: str
    command: str | list[str] | None = None
    script: str | None = None
    timeout: float = Field(default=30.0, gt=0.0)
    enabled: bool = True

    @field_validator("trigger", mode="before")
    @classmethod
    def normalize_trigger(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Trigger must be a string")
        v_clean = v.strip().lower().replace("-", "_").replace(" ", "_")
        if v_clean in ("on_reror", "on_error", "onerror", "on-reror", "on_reror"):
            return "on_error"
        if v_clean in ("before_agent", "beforeagent"):
            return "before_agent"
        if v_clean in ("after_agent", "afteragent"):
            return "after_agent"
        if v_clean in ("before_tool", "beforetool"):
            return "before_tool"
        if v_clean in ("after_tool", "aftertool"):
            return "after_tool"
        return v_clean

    @model_validator(mode="after")
    def validate_hook(self) -> HookConfig:
        if not self.name or not self.name.strip():
            raise ValueError("Hook name cannot be empty")

        valid_triggers = {"before_agent", "after_agent", "before_tool", "after_tool", "on_error"}
        if self.trigger not in valid_triggers:
            raise ValueError(f"Invalid trigger: {self.trigger}. Must be one of {sorted(valid_triggers)}")

        if self.command is not None and self.script is not None:
            raise ValueError("Cannot specify both command and script in a hook config")
        if self.command is None and self.script is None:
            raise ValueError("Must specify either command or script in a hook config")

        return self


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    cwd: Path = Field(default_factory=Path.cwd)
    hooks_enabled: bool = False
    hooks: list[HookConfig] = Field(default_factory=list)
    shell_env_policy: ShellEnvironmentPolicy = Field(default_factory=ShellEnvironmentPolicy)

    max_turns: int = 100
    retries: int = Field(default=3, ge=1)
    # Compact the history automatically when it uses more than this part of model.context_window.
    compaction_threshold: float = Field(default=0.8, gt=0.0, le=1.0)

    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)

    allowed_tools: list[str] | None = Field(
        None,
        description="If set, only these tools will be available to the agent",
    )

    developer_instructions: str | None = None
    user_instructions: str | None = None

    debug: bool = False
    approval: Annotated[ApprovalConfig, BeforeValidator(_coerce_approval)] = Field(default_factory=ApprovalConfig)

    @property
    def api_key(self) -> str:
        return _first_env(API_KEY_VARS[self.model.provider]) or ""

    @property
    def base_url(self) -> str:
        provider = self.model.provider
        return _first_env(BASE_URL_VARS[provider]) or DEFAULT_BASE_URLS[provider]

    @property
    def model_name(self) -> str:
        return self.model.name

    @model_name.setter
    def model_name(self, value: str) -> None:
        self.model.name = value

    @property
    def temperature(self) -> float | None:
        return self.model.temp

    @temperature.setter
    def temperature(self, value: float | None) -> None:
        self.model.temp = value

    def get_validation_errors(self) -> list[str]:
        errors: list[str] = []

        if not self.api_key:
            provider = self.model.provider.value
            var = API_KEY_VARS[self.model.provider][0]
            errors.append(
                f"No API key found for provider '{provider}'. Put {var}=... in .pair-programmer/.env "
                f"(project or home directory), or export {var}"
            )

        if not self.cwd.exists():
            errors.append(f"Working directory does not exist: {self.cwd}")

        return errors

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None
