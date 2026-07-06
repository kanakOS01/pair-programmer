from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field


class ModelConfig(BaseModel):
    name: str = "openai/gpt-oss-120b:free"
    temp: float = Field(default=1, ge=0.0, le=2.0)
    context_window: int = Field(256_000)


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
    policy: ApprovalPolicy = ApprovalPolicy.AUTO
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


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    cwd: Path = Field(default_factory=Path.cwd)
    shell_env_policy: ShellEnvironmentPolicy = Field(default_factory=ShellEnvironmentPolicy)

    max_turns: int = 100
    retries: int = 3

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
        return os.environ.get("API_KEY", "")

    @property
    def base_url(self) -> str:
        return os.environ.get("BASE_URL", "https://openrouter.ai/api/v1")

    @property
    def model_name(self) -> str:
        return self.model.name

    @model_name.setter
    def model_name(self, value: str) -> None:
        self.model.name = value

    @property
    def temperature(self) -> float:
        return self.model.temp

    @temperature.setter
    def temperature(self, value: float) -> None:
        self.model.temp = value

    def get_validation_errors(self) -> list[str]:
        errors: list[str] = []

        if not self.api_key:
            errors.append("No API key found. Set API_KEY environment variable")

        if not self.cwd.exists():
            errors.append(f"Working directory does not exist: {self.cwd}")

        return errors

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
