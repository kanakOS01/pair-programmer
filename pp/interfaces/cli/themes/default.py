from rich.theme import Theme


DEFAULT_THEME = Theme(
    {
        # core roles
        "user": "bold cyan",
        "assistant": "white",
        "system": "dim white",
        "agent": "bold magenta",

        # streaming / tokens
        "token": "white",
        "token.dim": "dim white",
        "delta": "bright_white",

        # structure
        "border": "dim",
        "panel": "dim",
        "header": "bold",
        "footer": "dim",

        # states
        "success": "bold green",
        "warning": "yellow",
        "error": "bold red",
        "debug": "dim cyan",

        # tool / execution
        "tool": "bold blue",
        "path": "underline cyan",
        "code": "bright_white",
        "diff.add": "green",
        "diff.remove": "red",
        "diff.change": "yellow",

        # tokens / usage
        "usage.prompt": "cyan",
        "usage.completion": "magenta",
        "usage.total": "bold white",

        # spinner / status
        "spinner": "bold cyan",
        "status": "dim white",
    }
)