from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from pp.config.config import Config
from pp.domain import ToolInvocation, ToolResult, ToolType
from pp.tools.base import Tool

logger = logging.getLogger(__name__)


class SubagentDefinition(BaseModel):
    name: str | None = None
    description: str | None = None
    goal_prompt: str
    allowed_tools: list[str] | None = None
    model: dict[str, Any] | None = None
    max_turns: int | None = None
    timeout_seconds: int | None = None


CODEBASE_INVESTIGATOR = SubagentDefinition(
    name="codebase_investigator",
    description="Investigates the codebase to answer questions about code structure, patterns, and implementations",
    goal_prompt=(
        "You are a codebase investigation specialist.\n"
        "Your job is to explore and understand code to answer questions.\n"
        "Use read_file, grep, glob, and list_dir to investigate.\n"
        "Do NOT modify any files."
    ),
    allowed_tools=["read_file", "grep", "glob", "list_dir"],
)

CODE_REVIEWER = SubagentDefinition(
    name="code_reviewer",
    description="Reviews code changes and provides feedback on quality, bugs, and improvements",
    goal_prompt=(
        "You are a code review specialist.\n"
        "Your job is to review code and provide constructive feedback.\n"
        "Look for bugs, code smells, security issues, and improvement opportunities.\n"
        "Use read_file, list_dir and grep to examine the code.\n"
        "Do NOT modify any files."
    ),
    allowed_tools=["read_file", "grep", "list_dir"],
    max_turns=10,
    timeout_seconds=300,
)


def get_default_subagent_definitions() -> list[SubagentDefinition]:
    return [
        CODEBASE_INVESTIGATOR,
        CODE_REVIEWER,
    ]


class SubagentParams(BaseModel):
    task: str = Field(..., description="The task description or question for the subagent to run.")


class SubagentTool(Tool):
    def __init__(self, name: str, subagent_config: SubagentDefinition, config: Config) -> None:
        super().__init__(config)
        self.name = f"subagent_{name}"
        self.description = subagent_config.description or f"Runs the subagent '{name}' with the given task."
        self.subagent_config = subagent_config
        self.schema = SubagentParams
        self.type = ToolType.READ

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        task = invocation.params.get("task")
        if not task:
            return ToolResult.error_result("No task provided")

        # Create a modified config for the subagent
        from pp.config.config import Config, ModelConfig

        model_cfg = self.config.model
        if self.subagent_config.model:
            model_cfg = ModelConfig(**self.subagent_config.model)

        subagent_config = Config(
            model=model_cfg,
            cwd=self.config.cwd,
            shell_env_policy=self.config.shell_env_policy,
            max_turns=self.subagent_config.max_turns or self.config.max_turns,
            allowed_tools=self.subagent_config.allowed_tools,
            developer_instructions=self.subagent_config.goal_prompt,
            user_instructions=None,
            debug=self.config.debug,
            hooks_enabled=self.config.hooks_enabled,
            hooks=self.config.hooks,
        )

        from pp.agents.coding import CodingAgent
        from pp.domain import AgentEventType

        final_response = None
        error_msg = None
        try:
            async with CodingAgent(config=subagent_config) as agent:
                async for event in agent.run(task):
                    if event.type == AgentEventType.TextComplete:
                        final_response = event.data.get("content")
                    elif event.type == AgentEventType.Done:
                        if not final_response:
                            final_response = event.data.get("response")
                    elif event.type == AgentEventType.Error:
                        error_msg = event.data.get("error")

            if error_msg:
                return ToolResult.error_result(f"Subagent failed: {error_msg}")
            elif final_response:
                return ToolResult.success_result(output=final_response)
            else:
                return ToolResult.error_result("Subagent completed but returned no response.")
        except Exception as e:
            logger.exception(f"Subagent {self.name} failed with error")
            return ToolResult.error_result(f"Subagent failed with error: {e}")
