import asyncio
import fnmatch
import os
import sys

from pydantic import BaseModel, Field

from pp.domain import ToolInvocation, ToolResult, ToolType
from pp.tools.base import Tool
from pp.utils.paths import resolve_path


class ShellToolParams(BaseModel):
    command: str = Field(..., description="The command to execute")
    timeout: int = Field(60, ge=1, le=600, description="Timeout in seconds")
    cwd: str | None = Field(None, description="Working directory for the command")


class ShellTool(Tool):
    name = "shell"
    description = "Execute a shell command"
    type = ToolType.WRITE
    schema = ShellToolParams

    MAX_OUTPUT_SIZE_BYTES = 1024 * 100  # 100 kb
    BLOCKED_COMMANDS = {
        "rm -rf /",
        "rm -rf ~",
        "rm -rf /*",
        "dd if=/dev/zero",
        "dd if=/dev/random",
        "mkfs",
        "fdisk",
        "parted",
        ":(){ :|:& };:",
        "chmod 777 /",
        "chmod -R 777",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "init 0",
        "init 6",
    }

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = ShellToolParams(**invocation.params)

        cmd = params.command.strip()

        for blocked_cmd in self.BLOCKED_COMMANDS:
            if blocked_cmd in cmd:
                return ToolResult.error_result(f"Command '{cmd}' contains blocked command '{blocked_cmd}'")

        cwd = invocation.cwd if params.cwd is None else resolve_path(invocation.cwd, params.cwd)
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
            process.kill()
            await process.wait()
            return ToolResult.error_result(f"Command '{cmd}' timed out after {params.timeout} seconds")

        stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

        output = ""
        if stdout_str:
            output += f"\n[STDOUT]\n{stdout_str.rstrip()}\n"
        if stderr_str:
            output += f"\n[STDERR]\n{stderr_str.rstrip()}\n"

        if process.returncode != 0:
            return ToolResult.error_result(
                f"Command '{cmd}' failed with exit code {process.returncode}\n{output}",
                exit_code=process.returncode,
            )

        if len(output) > self.MAX_OUTPUT_SIZE_BYTES:
            output = output[: self.MAX_OUTPUT_SIZE_BYTES] + "\n... [truncated]"

        return ToolResult.success_result(output=output, exit_code=process.returncode)

    def _build_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        shell_env_policy = self.config.shell_env_policy

        if not shell_env_policy.ignore_default_excludes:
            for pattern in shell_env_policy.exclude_patterns:
                env = {k: v for k, v in env.items() if not fnmatch.fnmatch(k.lower(), pattern.lower())}

        if shell_env_policy.set_vars:
            env.update(shell_env_policy.set_vars)

        return env
