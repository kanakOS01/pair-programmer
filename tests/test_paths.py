from pathlib import Path

import pytest

from pp.utils.paths import resolve_path_safe


@pytest.mark.parametrize("path", ["a.txt", "sub/a.txt", "sub/../a.txt", "."])
def test_paths_inside_base_are_allowed(project: Path, path: str):
    resolved = resolve_path_safe(project, path)
    assert resolved == (project / path).resolve()


@pytest.mark.parametrize("path", ["../x", "sub/../../x", "/etc/passwd", "~/.zshrc"])
def test_paths_outside_base_are_rejected(project: Path, path: str):
    with pytest.raises(ValueError, match="outside"):
        resolve_path_safe(project, path)


def test_absolute_path_inside_base_is_allowed(project: Path):
    assert resolve_path_safe(project, str(project / "a.txt")) == project / "a.txt"


def test_symlink_out_of_base_is_rejected(project: Path, tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / "link").symlink_to(outside)

    with pytest.raises(ValueError):
        resolve_path_safe(project, "link/secret.txt")
