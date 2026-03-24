from pathlib import Path


def resolve_path(base: str | Path, path: str | Path) -> Path:
    path = Path(path)

    if path.is_absolute():
        return path.resolve()

    return Path(base).resolve() / path


def is_binary_file(path: str | Path) -> bool:
    path = Path(path)

    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
            return b"\x00" in chunk
    except (OSError, IOError):
        return False


def resolve_path_safe(base: str | Path, path: str | Path) -> Path:
    base = Path(base).resolve()
    path = Path(path)

    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = base / path

    try:
        resolved.relative_to(base)
    except ValueError as _:
        raise ValueError("Access outside base directory not allowed") from None

    return resolved
