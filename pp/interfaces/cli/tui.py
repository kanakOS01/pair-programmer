from rich.theme import Theme
from rich.console import Console

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


    def stream_start(self) -> None:
        ...


    def stream_delta(self, content: str) -> None:
        self.console.print(content, end="", markup=False)


    def stream_end(self) -> None:
        ...
