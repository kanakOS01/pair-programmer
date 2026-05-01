import re
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from pp.config import Config
from pp.interfaces.cli.themes import DEFAULT_THEME
from pp.utils.text import truncate_text

# singleton console
_console: Console | None = None


def get_console(theme: Theme | None = None) -> Console:
    global _console
    if _console is None:
        _console = Console(theme=theme or DEFAULT_THEME, highlight=False)
    return _console


class TUI:
    def __init__(self, config: Config, console: Console | None = None):
        self.console = console or get_console()
        self.config = config
        self._assistant_stream_enabled = False
        self._tool_args_by_call_id: dict[str, dict[str, Any]] = {}
        self._max_block_tokens = 2500

    def stream_start(self) -> None:
        self.console.print()
        self.console.print(Rule(Text("Assistant", style="assistant"), style="assistant"))
        self._assistant_stream_enabled = True

    def stream_delta(self, content: str) -> None:
        self.console.print(content, end="", markup=False)

    def stream_end(self) -> None:
        if self._assistant_stream_enabled:
            self.console.print()

        self._assistant_stream_enabled = False

    def status(self, message: str) -> None: ...

    def system(self, message: str) -> None: ...

    def error(self, error: str) -> None:
        self.console.print()
        self.console.print(Panel(error, style="error"))

    def header(self, message: str) -> None: ...

    def table(self) -> None: ...

    def prompt(self, message: str) -> None: ...

    def code(self, code: str, language: str = "python") -> None:
        syntax = Syntax(code, language)
        self.console.print(syntax)

    def welcome(self, title: str, lines: list[str]) -> None:
        body = "\n".join(lines)
        self.console.print()
        self.console.print(
            Panel(
                Text(body, style="code"),
                title=Text(title, style="highlight"),
                title_align="left",
                border_style="border",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    def tool_call_start(self, *, call_id: str, name: str, tool_type: str | None, args: dict[str, Any]) -> None:
        self._tool_args_by_call_id[call_id] = args
        border_style = f"tool.{name}" if tool_type else "tool"

        title = Text.assemble(
            ("⏺ ", "muted"),
            (name, "tool"),
            ("  ", "muted"),
            (f"#{call_id[:8]}", "muted"),
        )
        panel = Panel(
            self._render_args_table(name, args) if args else Text("<no args>", style="italic muted"),
            title=title,
            title_align="left",
            subtitle=Text("running", style="muted"),
            subtitle_align="right",
            border_style=border_style,
            box=box.ROUNDED,
            padding=(1, 2),
        )
        self.console.print()
        self.console.print(panel)

    def tool_call_done(
        self,
        *,
        call_id: str,
        name: str,
        tool_type: str | None,
        success: bool,
        output: str,
        error: str | None,
        truncated: bool,
        diff: str | None,
        exit_code: int | None,
        meta: dict[str, Any] | None,
    ) -> None:
        border_style = f"tool.{name}" if tool_type else "tool"
        status_icon = "✔" if success else "✗"
        status_style = "success" if success else "error"
        blocks: list[RenderableType] = []

        args = self._tool_args_by_call_id.get(call_id) or {}

        path = None
        if isinstance(meta, dict) and isinstance(meta.get("path"), str):
            path = meta.get("path")

        if name == "read_file" and success:
            data = self._extract_read_file_code(output)
            if data:
                start_line, code = data

                start_line, end_line, total_lines = None, None, None
                if isinstance(meta, dict):
                    start_line = meta.get("start_line")
                    end_line = meta.get("end_line")
                    total_lines = meta.get("total_lines")

                language = self._guess_language(path)

                # blocks.append(Text())

                header_parts = [path, "•"] if path else []
                if start_line is not None and end_line is not None and total_lines is not None:
                    header_parts.append(f"Lines {start_line}-{end_line} of {total_lines}")

                blocks.append(Text(" ".join(header_parts), style="muted"))

                code_display = truncate_text(code, self.config.model_name, max_tokens=self._max_block_tokens)
                blocks.append(
                    Syntax(
                        code_display,
                        language,
                        theme="nord",
                        line_numbers=True,
                        start_line=start_line or 1,
                        word_wrap=False,
                    )
                )
            else:
                output_display = truncate_text(output, self.config.model_name, max_tokens=self._max_block_tokens)
                blocks.append(Text(output_display))

        elif name in ("write_file", "edit_file") and success and diff:
            output_line = output.strip() if output.strip() else Text("<no output>", style="muted")
            blocks.append(output_line)

            diff_display = truncate_text(diff, self.config.model_name, max_tokens=self._max_block_tokens)
            blocks.append(Syntax(diff_display, lexer="diff", theme="nord"))

        elif name == "shell" and success:
            cmd = args.get("command")
            if isinstance(cmd, str) and cmd.strip():
                blocks.append(Text(f"$ {cmd.strip()}", style="muted"))

            if exit_code is not None:
                blocks.append(Text(f"Exit code: {exit_code}", style="muted"))

            output_display = truncate_text(output, self.config.model_name, max_tokens=self._max_block_tokens)
            blocks.append(Syntax(output_display, lexer="zsh", theme="nord", word_wrap=True))

        if truncated:
            blocks.append(Text("\nNOTE: output truncated...", style="warning"))

        title = Text.assemble(
            (f"{status_icon} ", status_style),
            (name, "tool"),
            ("  ", "muted"),
            (f"#{call_id[:8]}", "muted"),
        )
        panel = Panel(
            Group(*blocks) if success else Text(error or ""),
            title=title,
            title_align="left",
            subtitle=Text("done" if success else "failed", style=status_style),
            subtitle_align="right",
            border_style=border_style,
            box=box.ROUNDED,
            padding=(1, 2),
        )
        # self.console.print()
        self.console.print(panel)

    def _render_args_table(self, tool_name: str, args: dict[str, Any]) -> Table:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="muted", justify="left", no_wrap=True)
        table.add_column(style="code", overflow="fold")

        for k, v in self._ordered_args(tool_name, args):
            if isinstance(v, str) and k in {"content", "old_str", "new_str"}:
                line_count = len(v.splitlines())
                byte_count = len(v.encode("utf-8", errors="replace"))
                v = f"{line_count} line(s) • ({byte_count} bytes)"

            if not isinstance(v, str):
                v = str(v)
            table.add_row(k, v)
        return table

    def _ordered_args(self, tool_name: str, args: dict[str, Any]) -> list[tuple[str, Any]]:
        _PREFERRED_ORDER = {
            "read_file": ["path", "offset", "limit"],
            "write_file": ["path", "create_dirs", "content"],
            "edit_file": ["path", "replace_all", "old_str", "new_str"],
            "shell": ["command", "timeout", "cwd"],
        }

        preferred = _PREFERRED_ORDER.get(tool_name, [])
        ordered: list[tuple[str, Any]] = []
        seen = set()

        for k in preferred:
            if k in args:
                ordered.append((k, args[k]))
                seen.add(k)

        remaining = set(args.keys() - seen)
        ordered.extend((key, args[key]) for key in (remaining))

        return ordered

    def _extract_read_file_code(self, text: str) -> tuple[int, str] | None:
        code_lines: list[str] = []
        start_line: int | None = None

        for line in text.splitlines():
            m = re.match(r"^\s*(\d+) \| (.*)$", line)
            if m:
                line_no = int(m.group(1))
                if start_line is None:
                    start_line = line_no
                code_lines.append(m.group(2))
            else:
                pass

        if start_line is None:
            return None

        return start_line, "\n".join(code_lines)

    def _guess_language(self, path: str | None) -> str:
        if not path:
            return "text"

        suffix = Path(path).suffix.lower()
        return {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "jsx",
            ".ts": "typescript",
            ".tsx": "tsx",
            ".json": "json",
            ".toml": "toml",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".md": "markdown",
            ".sh": "bash",
            ".bash": "bash",
            ".zsh": "bash",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".kt": "kotlin",
            ".swift": "swift",
            ".c": "c",
            ".h": "c",
            ".cpp": "cpp",
            ".hpp": "cpp",
            ".css": "css",
            ".html": "html",
            ".xml": "xml",
            ".sql": "sql",
        }.get(suffix, "text")
