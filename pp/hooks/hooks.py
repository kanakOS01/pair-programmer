import asyncio
import logging
import os
import shlex
from typing import Any, Optional

from pp.config.config import Config, HookConfig

logger = logging.getLogger(__name__)


class HookSystem:
    def __init__(self, config: Config):
        self.config = config

    async def run_hooks(self, trigger: str, context: Optional[dict[str, Any]] = None) -> None:
        """
        Run all enabled hooks for the given trigger.
        """
        if not self.config.hooks_enabled:
            return

        # Find matching enabled hooks
        hooks_to_run = [hook for hook in self.config.hooks if hook.enabled and hook.trigger == trigger]

        if not hooks_to_run:
            return

        logger.debug(f"Running {len(hooks_to_run)} hooks for trigger '{trigger}'")

        # Prepare environment variables with context
        env = os.environ.copy()
        if context:
            for k, v in context.items():
                # Prefix environment variables to avoid collisions and make it clear
                # e.g., if context has key 'agent_prompt', we set 'PP_AGENT_PROMPT'
                env[f"PP_{k.upper()}"] = str(v)

        for hook in hooks_to_run:
            env["PP_HOOK_NAME"] = hook.name
            env["PP_HOOK_TRIGGER"] = hook.trigger

            try:
                await self._execute_hook(hook, env)
            except Exception as e:
                logger.error(f"Error running hook '{hook.name}': {e}")

    async def _execute_hook(self, hook: HookConfig, env: dict[str, str]) -> None:
        """
        Execute a single hook subprocess with timeout.
        """
        logger.info(f"Executing hook '{hook.name}' (trigger: '{hook.trigger}')")

        proc = None
        cwd = self.config.cwd

        try:
            if hook.script is not None:
                # Run shell script
                proc = await asyncio.create_subprocess_shell(
                    hook.script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=cwd,
                )
            elif hook.command is not None:
                # Run command
                if isinstance(hook.command, list):
                    args = hook.command
                else:
                    args = shlex.split(hook.command)

                if not args:
                    raise ValueError(f"Empty command in hook '{hook.name}'")

                proc = await asyncio.create_subprocess_exec(
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=cwd,
                )
            else:
                return

            # Wait for execution with timeout
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=hook.timeout)
                exit_code = proc.returncode

                if exit_code != 0:
                    logger.warning(
                        f"Hook '{hook.name}' exited with non-zero code {exit_code}.\n"
                        f"STDOUT: {stdout.decode(errors='replace')}\n"
                        f"STDERR: {stderr.decode(errors='replace')}"
                    )
                else:
                    logger.debug(f"Hook '{hook.name}' executed successfully.\n" f"STDOUT: {stdout.decode(errors='replace')}")
            except asyncio.TimeoutError:
                # Kill process on timeout
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
                raise TimeoutError(f"Hook '{hook.name}' timed out after {hook.timeout} seconds") from None

        except Exception as e:
            logger.error(f"Failed to execute hook '{hook.name}': {e}")
            raise
