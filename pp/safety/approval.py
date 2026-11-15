from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from pp.config.config import ApprovalPolicy, Config
from pp.domain import ToolConfirmation

if TYPE_CHECKING:
    from pp.tools.base import Tool

logger = logging.getLogger(__name__)


class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_CONFIRMATION = "needs_confirmation"


@dataclass
class ApprovalContext:
    tool_name: str
    params: dict[str, Any] = field(default_factory=dict)
    mutability: bool = False
    affected_paths: list[Path] = field(default_factory=list)
    command: str | None = None
    is_danger: bool = False
    tool_type: str | None = None


DANGEROUS_COMMANDS = {
    "rm",
    "rmdir",
    "sudo",
    "su",
    "mv",
    "chmod",
    "chown",
    "dd",
    "mkfs",
    "shutdown",
    "reboot",
    "kill",
    "pkill",
    "killall",
    "truncate",
    "shred",
}

# A word in a shell command: a run of characters that are not whitespace, quotes or shell operators.
_COMMAND_WORD = re.compile(r"[^\s;&|()<>`'\"$]+")


def _is_dangerous_command(command: str) -> bool:
    """Return True if any word in the command is a dangerous command name.

    Every word is checked, not only the first one, so "cd x && rm -rf ." and
    "bash -c 'rm -rf .'" are also found. A false positive only asks for confirmation.
    """
    for word in _COMMAND_WORD.findall(command):
        if Path(word).name in DANGEROUS_COMMANDS:
            return True
    return False


def _resolve(path: Path | str, base: Path) -> Path:
    try:
        return (base / Path(path).expanduser()).resolve()
    except Exception:
        return base / Path(path)


def _is_path_safe(path: Path, safe_paths: list[str], base: Path) -> bool:
    resolved = _resolve(path, base)
    for safe in safe_paths:
        try:
            safe_p = _resolve(safe, base)
            if resolved == safe_p or safe_p in resolved.parents:
                return True
        except Exception:
            continue
    return False


class ApprovalChecker:
    def __init__(self, config: Config) -> None:
        self.config = config

    def check(self, context: ApprovalContext) -> ApprovalDecision:
        policy = self.config.approval.policy

        if policy == ApprovalPolicy.YOLO:
            return ApprovalDecision.APPROVED
        if policy == ApprovalPolicy.NEVER:
            return ApprovalDecision.REJECTED

        # 1. Check if tool is explicitly safe
        if context.tool_name in self.config.approval.safe_tools:
            return ApprovalDecision.APPROVED

        # 2. Check if paths are all safe
        paths_safe = False
        if self.config.approval.safe_paths and context.affected_paths:
            if all(_is_path_safe(p, self.config.approval.safe_paths, self.config.cwd) for p in context.affected_paths):
                paths_safe = True

        # 3. Check if context is flagged as danger
        if context.is_danger:
            return ApprovalDecision.NEEDS_CONFIRMATION

        # 4. Check dangerous commands
        if context.command and self.config.approval.dangerous_commands:
            if any(cmd in context.command for cmd in self.config.approval.dangerous_commands):
                return ApprovalDecision.NEEDS_CONFIRMATION

        # 5. Check if tool is in requires_approval_tools
        if context.tool_name in self.config.approval.requires_approval_tools:
            return ApprovalDecision.NEEDS_CONFIRMATION

        # 6. Check if tool type matches requires_approval_types
        if context.tool_type and context.tool_type in self.config.approval.requires_approval_types:
            return ApprovalDecision.NEEDS_CONFIRMATION

        # If paths are all safe and did not trigger explicit danger checks, approve it
        if paths_safe:
            return ApprovalDecision.APPROVED

        # 7. Check ON_REQUEST policy (requires confirmation for mutating tools)
        if policy == ApprovalPolicy.ON_REQUEST and context.mutability:
            return ApprovalDecision.NEEDS_CONFIRMATION

        # 8. Check AUTO_EDIT policy
        if policy == ApprovalPolicy.AUTO_EDIT:
            # AUTO_EDIT allows automatic edits (writing files), but shell, network, and mcp commands require confirmation.
            if context.tool_type in ("shell", "network", "mcp") or context.command is not None:
                return ApprovalDecision.NEEDS_CONFIRMATION
            return ApprovalDecision.APPROVED

        # Default fallback for AUTO / ON_FAILURE / others
        return ApprovalDecision.APPROVED


class ApprovalManager:
    def __init__(
        self,
        config: Config,
        confirmation_callback: Callable[[ToolConfirmation], bool | Awaitable[bool]] | None = None,
    ) -> None:
        self.config = config
        self.confirmation_callback = confirmation_callback
        self.checker = ApprovalChecker(config)

    def create_context(self, tool: Tool, params: dict[str, Any]) -> ApprovalContext:
        command = params.get("CommandLine") or params.get("command")
        mutability = tool.is_mutating()
        tool_type = tool.type.value if hasattr(tool, "type") and hasattr(tool.type, "value") else None

        affected_paths: list[Path] = []
        for key in ["TargetFile", "AbsolutePath", "DirectoryPath", "SearchPath", "path"]:
            if key in params:
                val = params[key]
                if isinstance(val, (str, Path)):
                    affected_paths.append(Path(val))
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, (str, Path)):
                            affected_paths.append(Path(item))

        # Check for dangerous command tokens
        is_danger = bool(command) and _is_dangerous_command(str(command))

        return ApprovalContext(
            tool_name=tool.name,
            params=params,
            mutability=mutability,
            affected_paths=affected_paths,
            command=command,
            is_danger=is_danger,
            tool_type=tool_type,
        )

    async def request_confirmation(self, confirmation: ToolConfirmation) -> bool:
        if self.confirmation_callback is None:
            return True

        res = self.confirmation_callback(confirmation)
        if inspect.isawaitable(res):
            return await res
        return res
