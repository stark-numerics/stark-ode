from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

AlgebraistFramePathLike = str | Sequence[str]


@dataclass(frozen=True, slots=True)
class AlgebraistFramePath:
    """Validated attribute path in an Algebraist frame."""

    parts: tuple[str, ...]

    @classmethod
    def from_value(cls, value: AlgebraistFramePathLike) -> "AlgebraistFramePath":
        """Build a validated frame path from a dotted string or sequence."""
        return cls(value)

    def __init__(self, value: AlgebraistFramePathLike) -> None:
        object.__setattr__(self, "parts", self._normalize(value))

    @staticmethod
    def _normalize(value: AlgebraistFramePathLike) -> tuple[str, ...]:
        if isinstance(value, str):
            parts = tuple(value.split("."))
        else:
            parts = tuple(value)

        if not parts:
            raise ValueError("AlgebraistFramePath cannot be empty.")
        for part in parts:
            if not isinstance(part, str):
                raise TypeError("AlgebraistFramePath parts must be strings.")
            if not part:
                raise ValueError("AlgebraistFramePath parts cannot be empty.")
            if not part.isidentifier():
                raise ValueError(
                    f"AlgebraistFramePath part {part!r} is not a valid identifier."
                )
        return parts

    @property
    def name(self) -> str:
        return "_".join(self.parts)

    def expression(self, root: str) -> str:
        if not root.isidentifier():
            raise ValueError(f"root {root!r} is not a valid identifier.")
        return ".".join((root, *self.parts))

    def __call__(self, root: object) -> Any:
        """Return the value reached by following this attribute path from `root`."""

        value = root
        for part in self.parts:
            value = getattr(value, part)
        return value

    def assign(self, root: object, value: Any) -> None:
        """Assign `value` at this path, creating missing parent objects."""

        target = self.ensure_parent(root)
        setattr(target, self.parts[-1], value)

    def ensure_parent(self, root: object) -> object:
        """Return this path's parent object, creating missing parents as needed."""

        target = root
        for part in self.parts[:-1]:
            child = getattr(target, part, None)
            if child is None:
                child = SimpleNamespace()
                setattr(target, part, child)
            target = child
        return target

    def __iter__(self):
        return iter(self.parts)

    def __len__(self) -> int:
        return len(self.parts)

    def __str__(self) -> str:
        return ".".join(self.parts)
