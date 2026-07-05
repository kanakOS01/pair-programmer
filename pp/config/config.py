from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


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


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    cwd: Path = Field(default_factory=Path.cwd)
    shell_env_policy: ShellEnvironmentPolicy = Field(default_factory=ShellEnvironmentPolicy)

    max_turns: int = 100
    retries: int = 3

    allowed_tools: list[str] | None = Field(
        None,
        description="If set, only these tools will be available to the agent",
    )

    subagents: dict[str, Any] = Field(default_factory=dict)

    developer_instructions: str | None = None
    user_instructions: str | None = None

    debug: bool = False

    @model_validator(mode="before")
    @classmethod
    def merge_default_subagents(cls, data: Any) -> Any:
        if isinstance(data, dict):
            from pp.tools.subagent import SubagentDefinition, get_default_subagent_definitions

            subagents = data.get("subagents", {})
            merged = {}
            for default_cfg in get_default_subagent_definitions():
                name = default_cfg.name
                if name in subagents:
                    user_subagent = subagents[name]
                    if isinstance(user_subagent, dict):
                        # Merge dictionaries, overriding default fields
                        merged_subagent = default_cfg.model_dump()
                        for k, v in user_subagent.items():
                            if v is not None:
                                merged_subagent[k] = v
                        merged[name] = SubagentDefinition.model_validate(merged_subagent)
                    else:
                        merged[name] = (
                            user_subagent
                            if isinstance(user_subagent, SubagentDefinition)
                            else SubagentDefinition.model_validate(user_subagent)
                        )
                else:
                    merged[name] = default_cfg

            # Add any other user defined subagents
            for name, user_subagent in subagents.items():
                if name not in merged:
                    if isinstance(user_subagent, dict) and "name" not in user_subagent:
                        user_subagent = user_subagent.copy()
                        user_subagent["name"] = name
                    merged[name] = (
                        user_subagent
                        if isinstance(user_subagent, SubagentDefinition)
                        else SubagentDefinition.model_validate(user_subagent)
                    )

            data["subagents"] = merged
        return data

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
