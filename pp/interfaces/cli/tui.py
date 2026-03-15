from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
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
