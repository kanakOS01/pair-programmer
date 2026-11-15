from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Awaitable, Callable

from pp.config import Config
from pp.domain import ToolCall, ToolConfirmation, ToolInvocation, ToolResult
from pp.hooks import HookSystem
from pp.safety import ApprovalChecker, ApprovalDecision, approval_context
from pp.tools.base import Tool
from pp.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Asks a person to allow one tool call. Returns True to allow it.
ConfirmFn = Callable[[ToolConfirmation], bool | Awaitable[bool]]


class ToolExecutor:
    """Runs one tool call from start to end: lookup, validation, approval, confirmation, hooks and the tool itself.

    run() never raises. Every outcome, also a rejection or a crash in the tool, is a ToolResult.
    If a call needs confirmation and confirm is None, the call is rejected.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        config: Config,
        hooks: HookSystem,
        confirm: ConfirmFn | None = None,
    ) -> None:
        self.registry = registry
        self.config = config
        self.hooks = hooks
        self.confirm = confirm
        self._checker = ApprovalChecker(config)

    async def run(self, call: ToolCall) -> ToolResult:
        tool = self.registry.get(call.name)
        if tool is None:
            return ToolResult.error_result(f"Tool {call.name} not found", metadata={"tool_name": call.name})

        validation_errors = tool.validate_params(call.args)
        if validation_errors:
            return ToolResult.error_result(
                f"Invalid parameters: {'; '.join(validation_errors)}",
                metadata={"tool_name": call.name, "validate_errors": validation_errors},
            )

        invocation = ToolInvocation(cwd=self.config.cwd, params=call.args)

        rejection = await self._approve(tool, invocation)
        if rejection is not None:
            return rejection

        hook_context = {"tool_name": call.name, "tool_args": json.dumps(call.args)}
        await self.hooks.run_hooks("before_tool", hook_context)

        try:
            result = await tool.execute(invocation)
        except Exception as e:
            logger.exception(f"Tool {call.name} raised unexpected error")
            result = ToolResult.error_result(f"Internal error: {e}", metadata={"tool_name": call.name})

        await self.hooks.run_hooks(
            "after_tool",
            {**hook_context, "tool_result": result.to_model_output(), "tool_success": str(result.ok)},
        )
        return result

    async def _approve(self, tool: Tool, invocation: ToolInvocation) -> ToolResult | None:
        """Return None if the call may run, or the rejection result."""
        try:
            effects = tool.effects(invocation.params)
        except Exception as e:
            logger.exception(f"Tool {tool.name} failed to declare its effects")
            return ToolResult.error_result(f"Internal error: {e}", metadata={"tool_name": tool.name})

        decision = self._checker.check(approval_context(tool, effects))

        if decision == ApprovalDecision.REJECTED:
            return ToolResult.error_result(
                "Tool execution rejected by approval policy",
                metadata={"tool_name": tool.name, "approval_decision": decision.value},
            )

        if decision == ApprovalDecision.NEEDS_CONFIRMATION:
            if self.confirm is None:
                return ToolResult.error_result(
                    "Tool execution needs confirmation, but no one can confirm it here",
                    metadata={"tool_name": tool.name, "approval_decision": "no_confirmation"},
                )

            confirmation = await tool.get_confirmation(invocation) or ToolConfirmation(
                tool_name=tool.name, params=invocation.params, description=f"Execute {tool.name}"
            )
            approved = self.confirm(confirmation)
            if inspect.isawaitable(approved):
                approved = await approved
            if not approved:
                return ToolResult.error_result(
                    "Tool execution rejected by user",
                    metadata={"tool_name": tool.name, "approval_decision": "rejected_by_user"},
                )

        return None
