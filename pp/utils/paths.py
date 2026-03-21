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
