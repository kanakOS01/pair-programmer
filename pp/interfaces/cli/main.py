import asyncio
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Literal

import typer

from pp.agents import CodingAgent
from pp.config import Config, load_config
from pp.context import CompactionLayer
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

    async def run(self, run_type: Literal["one_time", "interactive"] = "interactive", prompt: str = ""):
        if run_type == "one_time":
            return await self._run_once(prompt)
        elif run_type == "interactive":
            return await self._run_interactive()

    async def _run_once(self, prompt: str):
        async with CodingAgent(config=self.config) as agent:
            self.coding_agent = agent
            agent.session.approval_manager.confirmation_callback = self._confirm_tool_call
            return await self._process_message(prompt)

    async def _run_interactive(self):
        async with CodingAgent(config=self.config) as agent:
            self.coding_agent = agent
            agent.session.approval_manager.confirmation_callback = self._confirm_tool_call

            self.tui.welcome(
                title="Pair Programmer",
                lines=[
                    f"Model: {self.config.model_name}",
                    f"cwd: {self.config.cwd}",
                    "commands: /help /config /approval /model /mcp /exit",
                ],
            )

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
                        await self._process_message(inp)

                except KeyboardInterrupt:
                    console.print("\n[dim]Use /exit to exit standard mode.[/dim]")
                except EOFError:
                    break

        return None

    async def _process_message(self, message: str) -> str | None:
        if not self.coding_agent:
            return None

        assistant_streaming = False

        async for event in self.coding_agent.run(message):
            if event.type == AgentEventType.TextDelta:
                if not assistant_streaming:
                    self.tui.stream_start()
                    assistant_streaming = True

                content = event.data.get("content", "")
                self.tui.stream_delta(content)

            elif event.type == AgentEventType.TextComplete:
                final_response = event.data.get("content", "")
                if assistant_streaming:
                    self.tui.stream_end()
                    assistant_streaming = False

                return final_response

            elif event.type == AgentEventType.Error:
                error = event.data.get("error", "Unkown error")
                self.tui.error(error)

            elif event.type == AgentEventType.ToolCallStart:
                tool_name = event.data.get("name", "Unknown")
                tool = self.coding_agent.session.tool_registry.get(tool_name)
                tool_type = tool.type.value if tool else None

                self.tui.tool_call_start(
                    call_id=event.data.get("call_id", ""), name=tool_name, tool_type=tool_type, args=event.data.get("args", {})
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
        elif cmd == "/compact":
            await self._handle_compact_command(args)
        else:
            self.tui.show_command_error(f"Unknown command: {cmd}. Type /help for assistance.")
        return False

    def _handle_mcp_command(self):
        if not self.coding_agent or not self.coding_agent.session:
            self.tui.show_config_error("Agent/Session not initialized.")
            return

        mcp_manager = self.coding_agent.session.mcp_manager
        servers_data = []

        for name, client in mcp_manager.clients.items():
            tools = []
            if client._connected:
                for tool in self.coding_agent.session.tool_registry.list_tools():
                    if hasattr(tool, "client_name") and tool.client_name == name:
                        tools.append(tool.name)

            servers_data.append(
                {
                    "name": name,
                    "connected": client._connected,
                    "command": f"{client.command} {' '.join(client.args)}",
                    "tools": tools,
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
            compaction_layer = CompactionLayer(
                self.coding_agent.session.context_manager,
                self.coding_agent.session.llm,
            )
            # Run compaction and await it
            summary, usage = await compaction_layer.compact(keep_last_n)
            if not summary:
                self.tui.show_compact_no_op()
            else:
                self.tui.show_compact_success(summary, usage)
        except Exception as e:
            self.tui.show_command_error(f"Compaction failed: {e}")

    def _handle_config_command(self, args: str):
        args = args.strip()
        if not args:
            config_data = [
                ("model.name", self.config.model.name, "str"),
                ("model.temp", self.config.model.temp, "float"),
                ("model.context_window", self.config.model.context_window, "int"),
                ("cwd", str(self.config.cwd), "Path"),
                ("max_turns", self.config.max_turns, "int"),
                ("retries", self.config.retries, "int"),
                ("debug", self.config.debug, "bool"),
                ("allowed_tools", self.config.allowed_tools, "list[str] | None"),
                (
                    "approval",
                    self.config.approval.value if hasattr(self.config.approval, "value") else str(self.config.approval),
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
                if self.coding_agent:
                    self.coding_agent.session.context_manager.update_system_prompt()
            else:
                self.tui.show_config_error(f"Failed to update configuration key: {key}")

    def _handle_model_command(self, args: str):
        args = args.strip()
        if not args:
            self.tui.show_model(self.config.model_name)
            return

        self.config.model_name = args
        self.tui.show_model_success(args)
        if self.coding_agent:
            self.coding_agent.session.context_manager.update_system_prompt()

    def _handle_approval_command(self, args: str):
        from pp.config.config import ApprovalPolicy

        args = args.strip()
        if not args:
            self.tui.show_approval(
                self.config.approval.value if hasattr(self.config.approval, "value") else str(self.config.approval),
                [p.value for p in ApprovalPolicy],
            )
            return

        matched = None
        for p in ApprovalPolicy:
            if p.value.lower() == args.lower():
                matched = p
                break

        if matched:
            self.config.approval = matched
            self.tui.show_approval_success(matched.value)
            if self.coding_agent:
                self.coding_agent.session.context_manager.update_system_prompt()
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
        obj = self.config
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

        try:
            if expected_type is bool:
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
            elif expected_type is Path:
                val = Path(value_str)
            elif key == "allowed_tools":
                if value_str.lower() in ("none", "null", ""):
                    val = None
                else:
                    val = [t.strip() for t in value_str.split(",")]
            elif key == "approval":
                from pp.config.config import ApprovalPolicy

                matched = None
                for p in ApprovalPolicy:
                    if p.value.lower() == value_str.lower():
                        matched = p
                        break
                if matched:
                    val = matched
                else:
                    return False
            else:
                val = value_str

            setattr(obj, attr, val)

            if key == "allowed_tools":
                if self.coding_agent:
                    from pp.tools.registry import create_default_registry

                    self.coding_agent.session.tool_registry = create_default_registry(self.config)
                    self.coding_agent.session.context_manager.update_system_prompt(
                        self.coding_agent.session.tool_registry.list_tools()
                    )

            return True
        except Exception as e:
            self.tui.show_config_error(f"Type conversion error: {e}")
            return False


def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
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

    if prompt:
        res = await cli.run(run_type="one_time", prompt=prompt)
        if res is None:
            sys.exit(1)
    else:
        await cli.run()


if __name__ == "__main__":
    app()
