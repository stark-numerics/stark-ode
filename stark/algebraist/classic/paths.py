from __future__ import annotations

from collections.abc import Sequence

AlgebraistPath = tuple[str, ...]


def normalize_path(path: str | Sequence[str]) -> AlgebraistPath:
    if isinstance(path, str):
        parts = tuple(path.split("."))
    else:
        parts = tuple(path)

    if not parts:
        raise ValueError("Algebraist paths cannot be empty.")

    for part in parts:
        if not isinstance(part, str) or not part:
            raise ValueError(f"Invalid Algebraist path segment {part!r}.")
        if not part.isidentifier():
            raise ValueError(f"Algebraist path segment {part!r} is not a valid Python identifier.")

    return parts


def path_expression(root: str, path: AlgebraistPath) -> str:
    return ".".join((root, *path))
