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

from pp.interfaces.cli.themes import DEFAULT_THEME

# singleton console
_console: Console | None = None


def get_console(theme: Theme | None = None) -> Console:
    global _console
    if _console is None:
        _console = Console(theme=theme or DEFAULT_THEME, highlight=False)
    return _console


class TUI:
    def __init__(self, console: Console | None = None):
        self.console = console or get_console()
        self._assistant_stream_enabled = False
        self._tool_args_by_call_id: dict[str, dict[str, Any]] = {}

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

    def tool_call_start(self, *, call_id: str, name: str, tool_type: str | None, args: dict[str, Any]) -> None:
        self._tool_args_by_call_id[call_id] = args
        border_style = f"tool.{tool_type}" if tool_type else "tool"

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
        meta: dict[str, Any] | None,
        truncated: bool,
    ) -> None:
        border_style = f"tool.{tool_type}" if tool_type else "tool"
        status_icon = "✔" if success else "✗"
        status_style = "success" if success else "error"
        blocks: list[RenderableType] = []

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

                blocks.append(Text())

                header_parts = [path, "•"] if path else []
                if start_line is not None and end_line is not None and total_lines is not None:
                    header_parts.append(f"Lines {start_line}-{end_line} of {total_lines}")

                blocks.append(Text(" ".join(header_parts), style="muted"))

                blocks.append(
                    Syntax(
                        code,
                        language,
                        theme="nord",
                        line_numbers=True,
                        start_line=start_line or 1,
                        word_wrap=False,
                    )
                )

        if truncated:
            blocks.append(Text("NOTE: output truncated...", style="warning"))

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
        self.console.print()
        self.console.print(panel)

    def _render_args_table(self, tool_name: str, args: dict[str, Any]) -> Table:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="muted", justify="left", no_wrap=True)
        table.add_column(style="code", overflow="fold")

        for k, v in self._ordered_args(tool_name, args):
            table.add_row(k, v)
        return table

    def _ordered_args(self, tool_name: str, args: dict[str, Any]) -> list[tuple[str, Any]]:
        _PREFERRED_ORDER = {
            "read_file": ["path", "offset", "limit"],
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
        body = text
        header_match = re.match(r"^Showing lines (\d+)-(\d+) of  (\d+)\n\n", text)

        if header_match:
            body = text[header_match.end() :]

        code_lines: list[str] = []
        start_line: int | None = None

        for line in body.splitlines():
            m = re.match(r"^\s*(\d+) \| (.*)$", line)
            if not m:
                return None
            line_no = int(m.group(1))
            if start_line is None:
                start_line = line_no
            code_lines.append(m.group(2))

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
