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
        # tokens / usage
        "usage.prompt": "cyan",
        "usage.completion": "magenta",
        "usage.total": "bold white",
        # spinner / status
        "spinner": "bold cyan",
        "status": "dim white",
        "muted": "dim white",
        # general
        "info": "cyan",
        "warning": "yellow",
        "error": "bright_red bold",
        "success": "green",
        "dim": "dim",
        "highlight": "bold cyan",
        "debug": "dim cyan",
        # Tools
        "tool": "bright_magenta bold",
        "tool.read": "cyan",
        "tool.write": "yellow",
        "tool.shell": "magenta",
        "tool.network": "bright_blue",
        "tool.memory": "green",
        "tool.mcp": "bright_cyan",
        "path": "underline cyan",
        # Code / blocks
        "code": "white",
    }
)
