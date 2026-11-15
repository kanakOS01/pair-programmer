import asyncio
import fnmatch
import os
import re
import sys
from typing import Any

from pydantic import BaseModel, Field

from pp.domain import ToolEffects, ToolInvocation, ToolResult, ToolType
from pp.tools.base import Tool
from pp.utils.paths import resolve_path_safe
from pp.utils.process import signal_process_group


class ShellToolParams(BaseModel):
    command: str = Field(..., description="The command to execute")
    timeout: int = Field(60, ge=1, le=600, description="Timeout in seconds")
    cwd: str | None = Field(None, description="Working directory for the command")


class ShellTool(Tool):
    name = "shell"
    description = "Execute a shell command"
    type = ToolType.SHELL
    schema = ShellToolParams

    MAX_OUTPUT_SIZE_BYTES = 1024 * 100  # 100 kb
    # Best-effort blocklist. It does not replace the approval policy.
    # The patterns run on the command after quotes are removed and whitespace is collapsed.
    BLOCKED_PATTERNS = [
        re.compile(r"\brm\s+(-\S+\s+)*(/|/\*|~|~/|~/\*|\$HOME/?)(\s|$|[;&|])"),
        re.compile(r"\bdd\b.*\bif=/dev/(zero|random|urandom)\b"),
        re.compile(r"\bdd\b.*\bof=/dev/(sd|hd|nvme|disk)"),
        re.compile(r"\bmkfs(\.\w+)?\b"),
        re.compile(r"\b(fdisk|parted|diskutil\s+erase\w*)\b"),
        re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),
        re.compile(r"\bchmod\s+(-\S+\s+)*777\s+/(\s|$)"),
        re.compile(r"\bchmod\s+-R\s+777\b"),
        re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"),
        re.compile(r"\binit\s+[06]\b"),
    ]

    def effects(self, params: dict[str, Any]) -> ToolEffects:
        # No paths: a shell command can touch any file, so its cwd must not count as a safe path.
        return ToolEffects(command=params["command"])

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = ShellToolParams(**invocation.params)

        cmd = params.command.strip()

        blocked = self._find_blocked_pattern(cmd)
        if blocked:
            return ToolResult.error_result(f"Command '{cmd}' matches blocked pattern '{blocked}'")

        try:
            cwd = invocation.cwd if params.cwd is None else resolve_path_safe(invocation.cwd, params.cwd)
        except ValueError as e:
            return ToolResult.error_result(str(e))
        if not cwd.exists():
            return ToolResult.error_result(f"Working directory '{cwd}' does not exist")

        env = self._build_environment()
        if sys.platform == "win32":
            shell_cmd = ["cmd.exe", "/c", params.command]
        else:
            shell_cmd = ["/bin/bash", "-c", params.command]

        process = await asyncio.create_subprocess_exec(
            *shell_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=params.timeout)
        except asyncio.TimeoutError:
            signal_process_group(process)
            await process.wait()
            return ToolResult.error_result(f"Command '{cmd}' timed out after {params.timeout} seconds")
        except asyncio.CancelledError:
            # The user stopped the turn (Ctrl-C). Stop the command too: it runs in its own process group,
            # so it would otherwise keep running.
            signal_process_group(process)
            raise

        stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

        output = ""
        if stdout_str:
            output += f"\n[STDOUT]\n{stdout_str.rstrip()}\n"
        if stderr_str:
            output += f"\n[STDERR]\n{stderr_str.rstrip()}\n"

        truncated = len(output) > self.MAX_OUTPUT_SIZE_BYTES
        if truncated:
            output = output[: self.MAX_OUTPUT_SIZE_BYTES] + "\n... [truncated]"

        if process.returncode != 0:
            return ToolResult.error_result(
                f"Command '{cmd}' failed with exit code {process.returncode}\n{output}",
                exit_code=process.returncode,
                truncated=truncated,
            )

        return ToolResult.success_result(output=output, exit_code=process.returncode, truncated=truncated)

    def _find_blocked_pattern(self, cmd: str) -> str | None:
        normalized = " ".join(cmd.replace('"', "").replace("'", "").replace("\\", "").split())
        for pattern in self.BLOCKED_PATTERNS:
            if pattern.search(normalized):
                return pattern.pattern
        return None

    def _build_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        shell_env_policy = self.config.shell_env_policy

        if not shell_env_policy.ignore_default_excludes:
            for pattern in shell_env_policy.exclude_patterns:
                env = {k: v for k, v in env.items() if not fnmatch.fnmatch(k.lower(), pattern.lower())}

        if shell_env_policy.set_vars:
            env.update(shell_env_policy.set_vars)

        return env
