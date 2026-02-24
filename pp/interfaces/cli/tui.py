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


