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


def create_dir(path: str | Path) -> Path:
    """
    Creates a directory and any necessary parent directories if they don't exist.

    Raises:
        NotADirectoryError: If the path exists but is not a directory.
    """
    path = Path(path)

    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"Path {path} is not a directory")
        return path

    path.mkdir(parents=True, exist_ok=True)
    return path


def create_parent_dir(path: str | Path) -> Path:
    """
    Creates the parent directory of a given path if it doesn't exist.

    Raises:
        NotADirectoryError: If the parent path exists but is not a directory.
    """
    return create_dir(Path(path).parent)


def get_config_dir() -> Path:
    return Path.home() / ".pair-programmer"


def get_data_dir() -> Path:
    return Path.home() / ".pair-programmer" / "user"
