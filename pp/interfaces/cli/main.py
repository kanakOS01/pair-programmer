import asyncio
import signal
import sys
from contextlib import AsyncExitStack, nullcontext
from functools import wraps
from pathlib import Path
from typing import Any, Literal

import typer

from pp.agents import CodingAgent
from pp.config import Config, load_config
from pp.domain import AgentEventType, ToolConfirmation
from pp.interfaces.cli.tui import TUI, get_console

app = typer.Typer()

console = get_console()


class CLI:
    def __init__(self, config: Config) -> None:
        self.coding_agent: CodingAgent | None = None
        self.config = config
        self.tui: TUI = TUI(config, console)

    def _confirm_tool_call(self, confirmation: ToolConfirmation) -> bool:
        # A live display (spinner or tool panel) redraws its area and can hide this question.
        self.tui.stop_live_displays()
        self.tui.console.print()
        self.tui.console.print("[warning]Requires Confirmation:[/warning]")
        self.tui.console.print(f"Tool: [info]{confirmation.tool_name}[/info]")
        self.tui.console.print(f"Action: {confirmation.description}")
        if confirmation.params:
            self.tui.console.print("Parameters:")
            for k, v in confirmation.params.items():
                self.tui.console.print(f"  - {k}: {v}")

        try:
            inp = self.tui.console.input("\n[bold]Approve? (y/N): [/bold]").strip().lower()
            return inp in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            return False

    async def run(
        self,
        run_type: Literal["one_time", "interactive"] = "interactive",
        prompt: str = "",
        session_id: str | None = None,
    ):
        if run_type == "one_time":
            return await self._run_once(prompt, session_id=session_id)
        elif run_type == "interactive":
            return await self._run_interactive(session_id=session_id)

    async def _start_agent(self, stack: AsyncExitStack) -> CodingAgent:
        """Start the agent and its MCP servers. Show a spinner while the servers start, and show any failures."""
        server_names = list(self.config.mcp_servers)
        with self.tui.mcp_starting(server_names) if server_names else nullcontext():
            agent = await stack.enter_async_context(CodingAgent(config=self.config))

        self.coding_agent = agent
        agent.session.tool_executor.confirm = self._confirm_tool_call
        return agent

    def _mcp_summary_lines(self) -> list[str]:
        if not self.coding_agent:
            return []
        manager = self.coding_agent.session.mcp_manager
        lines = []
        for name, client in manager.clients.items():
            if client.connected:
                count = sum(1 for t in self.coding_agent.session.mcp_tools if getattr(t, "client_name", None) == name)
                lines.append(f"MCP {name}: connected, {count} tools")
            else:
                lines.append(f"MCP {name}: failed (see below)")
        return lines

    async def _run_once(self, prompt: str, session_id: str | None = None):
        async with AsyncExitStack() as stack:
            agent = await self._start_agent(stack)
            self.tui.show_mcp_errors(agent.session.mcp_manager.errors)
            if session_id:
                try:
                    agent.session.load(session_id)
                except FileNotFoundError:
                    self.tui.error(f"Session '{session_id}' not found.")
                    return None
            return await self._run_turn(prompt)

    async def _run_interactive(self, session_id: str | None = None):
        async with AsyncExitStack() as stack:
            agent = await self._start_agent(stack)
            if session_id:
                try:
                    agent.session.load(session_id)
                    console.print(f"[success]Resumed session {session_id}[/success]")
                except FileNotFoundError:
                    self.tui.error(f"Session '{session_id}' not found.")
                    return None

            self.tui.welcome(
                title="Pair Programmer",
                lines=[
                    f"Model: {self.config.model_name} ({self.config.model.provider.value})",
                    f"cwd: {self.config.cwd}",
                    f"Session ID: {agent.session.session_id}",
                    *self._mcp_summary_lines(),
                    "commands: /help /config /approval /model /mcp /usage /resume /compact /exit",
                ],
            )
            self.tui.show_mcp_errors(agent.session.mcp_manager.errors)

            while True:
                try:
                    inp = console.input("\n[user]>[/user] ").strip()
                    if not inp:
                        continue

                    if inp.startswith("/"):
                        should_exit = await self._handle_command(inp)
                        if should_exit:
                            break
                    else:
                        await self._run_turn(inp)

                except KeyboardInterrupt:
                    console.print("\n[dim]Use /exit to exit standard mode.[/dim]")
                except EOFError:
                    break

        return None

    async def _run_turn(self, message: str) -> str | None:
        """Run one turn. Ctrl-C stops the turn and returns to the prompt; it does not exit pp."""
        task = asyncio.create_task(self._process_message(message))
        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, task.cancel)
        except NotImplementedError:
            pass  # Windows: no signal handlers on the event loop. Ctrl-C raises KeyboardInterrupt there.

        try:
            return await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise  # pp itself is being cancelled, not only this turn.
            self.tui.stop_live_displays()
            self.tui.console.print("\n[warning]Interrupted.[/warning] [dim]Type a new message, or /exit to quit.[/dim]")
            return None
        finally:
            try:
                # Restores the default handler, which raises KeyboardInterrupt at the prompt.
                loop.remove_signal_handler(signal.SIGINT)
            except NotImplementedError:
                pass

    async def _process_message(self, message: str) -> str | None:
        if not self.coding_agent:
            return None

        assistant_streaming = False
        final_response: str | None = None

        self.tui.thinking_start()
        try:
            async for event in self.coding_agent.run(message):
                # Stop the spinner before any output. Start and Done print nothing, so the spinner keeps running.
                # It starts again when the model works after a tool call.
                if event.type not in (AgentEventType.Start, AgentEventType.Done):
                    self.tui.thinking_stop()

                if event.type == AgentEventType.TextDelta:
                    if not assistant_streaming:
                        self.tui.stream_start()
                        assistant_streaming = True

                    content = event.data.get("content", "")
                    self.tui.stream_delta(content)

                elif event.type == AgentEventType.TextComplete:
                    # Do not return here: the same turn can also contain tool calls.
                    final_response = event.data.get("content", "")
                    if assistant_streaming:
                        self.tui.stream_end()
                        assistant_streaming = False

                elif event.type == AgentEventType.Done:
                    final_response = event.data.get("response") or final_response

                elif event.type == AgentEventType.Error:
                    error = event.data.get("error", "Unkown error")
                    self.tui.error(error)

                elif event.type == AgentEventType.ToolCallStart:
                    tool_name = event.data.get("name", "Unknown")
                    tool = self.coding_agent.session.tool_registry.get(tool_name)
                    tool_type = tool.type.value if tool else None

                    self.tui.tool_call_start(
                        call_id=event.data.get("call_id", ""),
                        name=tool_name,
                        tool_type=tool_type,
                        args=event.data.get("args", {}),
                    )

                elif event.type == AgentEventType.ToolCallDone:
                    tool_name = event.data.get("name", "Unknown")
                    tool = self.coding_agent.session.tool_registry.get(tool_name)
                    tool_type = tool.type.value if tool else None

                    self.tui.tool_call_done(
                        call_id=event.data.get("call_id", ""),
                        name=tool_name,
                        tool_type=tool_type,
                        success=event.data.get("success", False),
                        output=event.data.get("output", ""),
                        error=event.data.get("error", None),
                        diff=event.data.get("diff", None),
                        truncated=event.data.get("truncated", False),
                        exit_code=event.data.get("exit_code"),
                        meta=event.data.get("metadata", None),
                    )
                    # The model gets the tool result next and works on it.
                    self.tui.thinking_start()
        finally:
            self.tui.thinking_stop()

        if assistant_streaming:
            self.tui.stream_end()

        return final_response

    async def _handle_command(self, inp: str) -> bool:
        """Handles a slash command. Returns True if the CLI loop should exit."""
        parts = inp.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "/exit":
            self.tui.show_exit()
            return True
        elif cmd == "/help":
            self.tui.show_help()
        elif cmd == "/config":
            self._handle_config_command(args)
        elif cmd == "/approval":
            self._handle_approval_command(args)
        elif cmd == "/model":
            self._handle_model_command(args)
        elif cmd == "/mcp":
            self._handle_mcp_command()
        elif cmd == "/usage":
            if not self.coding_agent or not self.coding_agent.session:
                self.tui.show_config_error("Agent/Session not initialized.")
            else:
                self.tui.show_usage(self.coding_agent.session.token_usage)
        elif cmd == "/resume":
            await self._handle_resume_command(args)
        elif cmd == "/compact":
            await self._handle_compact_command(args)
        else:
            self.tui.show_command_error(f"Unknown command: {cmd}. Type /help for assistance.")
        return False

    async def _handle_resume_command(self, args: str):
        if not self.coding_agent or not self.coding_agent.session:
            self.tui.show_config_error("Agent/Session not initialized.")
            return

        session_id = args.strip()
        if not session_id:
            await self._handle_resume_interactive()
        else:
            try:
                self.coding_agent.session.load(session_id)
                console.print(f"[success]Resumed session {session_id}[/success]")
            except FileNotFoundError:
                self.tui.show_command_error(f"Session '{session_id}' not found.")

    async def _handle_resume_interactive(self):
        session_id = await self.select_session()
        if not session_id:
            return

        try:
            self.coding_agent.session.load(session_id)
            console.print(f"[success]Resumed session {session_id}[/success]")
        except FileNotFoundError:
            self.tui.show_command_error(f"Session '{session_id}' not found.")

    async def select_session(self) -> str | None:
        from pp.core.session import get_sessions

        all_sessions = get_sessions()
        if not all_sessions:
            console.print("[error]No saved sessions found.[/error]")
            return None

        current_page = 0
        page_size = 5

        while True:
            total_pages = (len(all_sessions) + page_size - 1) // page_size
            start_idx = current_page * page_size
            end_idx = start_idx + page_size
            page_sessions = all_sessions[start_idx:end_idx]

            self.tui.show_sessions(page_sessions, current_page, total_pages)

            prompt_lines = []
            if current_page + 1 < total_pages:
                prompt_lines.append("[bold]n[/bold] for Next page")
            if current_page > 0:
                prompt_lines.append("[bold]p[/bold] for Previous page")
            prompt_lines.append("number [bold]1-5[/bold] to resume a session")
            prompt_lines.append("Session ID to resume")
            prompt_lines.append("press [bold]Enter[/bold] to cancel")

            console.print(f"Options: {', '.join(prompt_lines)}")

            try:
                inp = console.input("\nChoose option: ").strip()
                if not inp:
                    return None

                if inp.lower() in ("n", "next") and current_page + 1 < total_pages:
                    current_page += 1
                    continue
                elif inp.lower() in ("p", "prev") and current_page > 0:
                    current_page -= 1
                    continue

                if inp.isdigit():
                    idx = int(inp) - 1
                    if 0 <= idx < len(page_sessions):
                        return page_sessions[idx]["session_id"]
                    else:
                        console.print("[error]Invalid index selected.[/error]")
                        continue

                uuid_matches = [s["session_id"] for s in all_sessions if s["session_id"] == inp]
                if uuid_matches:
                    return uuid_matches[0]
                else:
                    console.print(f"[error]Session '{inp}' not found in saved sessions.[/error]")

            except (KeyboardInterrupt, EOFError):
                return None

    def _handle_mcp_command(self):
        if not self.coding_agent or not self.coding_agent.session:
            self.tui.show_config_error("Agent/Session not initialized.")
            return

        mcp_manager = self.coding_agent.session.mcp_manager
        servers_data = []

        for name, client in mcp_manager.clients.items():
            tools = []
            if client.connected:
                for tool in self.coding_agent.session.tool_registry.list_tools():
                    if hasattr(tool, "client_name") and tool.client_name == name:
                        tools.append(tool.name)

            servers_data.append(
                {
                    "name": name,
                    "connected": client.connected,
                    "command": f"{client.command} {' '.join(client.args)}",
                    "tools": tools,
                    "error": mcp_manager.errors.get(name),
                }
            )

        self.tui.show_mcp_servers(servers_data)

    async def _handle_compact_command(self, args: str):
        if not self.coding_agent or not self.coding_agent.session:
            self.tui.show_config_error("Agent/Session not initialized.")
            return

        keep_last_n = 4
        args = args.strip()
        if args:
            try:
                keep_last_n = int(args)
            except ValueError:
                self.tui.show_command_error(f"Invalid keep_last_n: {args}. Must be an integer.")
                return

        self.tui.show_compact_start(keep_last_n)
        try:
            result = await self.coding_agent.session.compact(keep_last_n)
            if result is None:
                self.tui.show_compact_no_op()
            else:
                summary, usage = result
                self.tui.show_compact_success(summary, usage)
        except Exception as e:
            self.tui.show_command_error(f"Compaction failed: {e}")

    def _handle_config_command(self, args: str):
        args = args.strip()
        if not args:
            config_data = [
                ("model.provider", self.config.model.provider.value, "openai | openrouter"),
                ("model.name", self.config.model.name, "str"),
                ("model.temp", self.config.model.temp, "float | None"),
                ("model.reasoning_effort", self.config.model.reasoning_effort, "none | low | medium | high | xhigh | max"),
                ("model.context_window", self.config.model.context_window, "int"),
                ("cwd", str(self.config.cwd), "Path"),
                ("max_turns", self.config.max_turns, "int"),
                ("retries", self.config.retries, "int"),
                ("debug", self.config.debug, "bool"),
                ("allowed_tools", self.config.allowed_tools, "list[str] | None"),
                (
                    "approval",
                    self.config.approval.policy.value,
                    "ApprovalPolicy",
                ),
            ]
            self.tui.show_config(config_data)
            return

        parts = args.split(maxsplit=1)
        key = parts[0]
        if len(parts) == 1:
            val = self._get_config_attr(key)
            if val is None:
                self.tui.show_config_error(f"Invalid or unknown configuration key: {key}")
            else:
                self.tui.show_config_value(key, val)
        else:
            value_str = parts[1]
            success = self._set_config_attr(key, value_str)
            if success:
                new_val = self._get_config_attr(key)
                self.tui.show_config_success(key, new_val)
            else:
                self.tui.show_config_error(f"Failed to update configuration key: {key}")

    def _handle_model_command(self, args: str):
        args = args.strip()
        if not args:
            self.tui.show_model(self.config.model_name)
            return

        self.config.model_name = args
        self.tui.show_model_success(args)

    def _handle_approval_command(self, args: str):
        from pp.config.config import ApprovalPolicy

        args = args.strip()
        if not args:
            self.tui.show_approval(
                self.config.approval.policy.value,
                [p.value for p in ApprovalPolicy],
            )
            return

        matched = None
        for p in ApprovalPolicy:
            if p.value.lower() == args.lower():
                matched = p
                break

        if matched:
            self.config.approval.policy = matched
            self.tui.show_approval_success(matched.value)
        else:
            self.tui.show_command_error(
                f"Invalid approval policy: {args}. Must be one of: {', '.join(p.value for p in ApprovalPolicy)}"
            )

    def _get_config_attr(self, key: str) -> Any:
        parts = key.split(".")
        obj = self.config
        for p in parts:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                return None
        return obj

    def _set_config_attr(self, key: str, value_str: str) -> bool:
        parts = key.split(".")
        obj: Any = self.config
        for p in parts[:-1]:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                return False

        attr = parts[-1]
        if not hasattr(obj, attr):
            return False

        current_val = getattr(obj, attr)
        expected_type = type(current_val) if current_val is not None else str

        val: Any
        try:
            if value_str.lower() in ("none", "null") and key != "allowed_tools":
                # Clear an optional setting, for example model.temp. Pydantic rejects it for required settings.
                val = None
            elif expected_type is bool:
                if value_str.lower() in ("true", "1", "yes", "on"):
                    val = True
                elif value_str.lower() in ("false", "0", "no", "off"):
                    val = False
                else:
                    return False
            elif expected_type is int:
                val = int(value_str)
            elif expected_type is float:
                val = float(value_str)
            elif issubclass(expected_type, Path):
                val = Path(value_str)
            elif key == "allowed_tools":
                if value_str.lower() in ("none", "null", ""):
                    val = None
                else:
                    val = [t.strip() for t in value_str.split(",")]
            elif key in ("approval", "approval.policy"):
                from pp.config.config import ApprovalPolicy

                # Set the policy field. Do not replace the ApprovalConfig object.
                obj = self.config.approval
                attr = "policy"
                val = ApprovalPolicy(value_str.lower())
            else:
                val = value_str

            setattr(obj, attr, val)

            if key == "allowed_tools" and self.coding_agent:
                self.coding_agent.session.rebuild_tool_registry()

            return True
        except Exception as e:
            self.tui.show_config_error(f"Type conversion error: {e}")
            return False


def _raise_keyboard_interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        # asyncio.run() installs its own Ctrl-C handler, which cancels the whole program at the next await.
        # A different handler stops it from doing that. pp handles Ctrl-C itself: at the prompt it raises
        # KeyboardInterrupt, and during a turn _run_turn() stops only that turn.
        signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
        return asyncio.run(f(*args, **kwargs))

    return wrapper


@app.command()
@coro
async def main(
    prompt: str = "",
    cwd: Path = typer.Option(
        Path.cwd(),
        "--cwd",
        "-c",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Set the current working directory",
    ),
    resume: str = typer.Option(
        None,
        "--resume",
        "-r",
        help="Resume a previous session by session ID (UUID) or list them",
    ),
):
    """
    Entrypoint of pp (pair-programmer) CLI
    """
    try:
        config = load_config(cwd=cwd)
    except Exception as e:
        console.print(f"[error]Error loading config: {e}[/error]")
        sys.exit(1)

    errors = config.get_validation_errors()
    if errors:
        console.print("[error]Config validation errors:[/error]")
        for error in errors:
            console.print(f"- {error}")
        sys.exit(1)

    cli = CLI(config)

    session_id = None
    if resume is not None:
        session_id = resume.strip()
        if not session_id or session_id.lower() == "list":
            session_id = await cli.select_session()
            if not session_id:
                sys.exit(0)

    if prompt:
        res = await cli.run(run_type="one_time", prompt=prompt, session_id=session_id)
        if res is None:
            sys.exit(1)
    else:
        await cli.run(run_type="interactive", session_id=session_id)


if __name__ == "__main__":
    app()
