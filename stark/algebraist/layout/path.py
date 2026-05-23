from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

AlgebraistLayoutPathLike = str | Sequence[str]


@dataclass(frozen=True, slots=True)
class AlgebraistLayoutPath:
    """Validated attribute path in an Algebraist layout."""

    parts: tuple[str, ...]

    @classmethod
    def from_value(cls, value: AlgebraistLayoutPathLike) -> "AlgebraistLayoutPath":
        """Build a validated layout path from a dotted string or sequence."""
        return cls(value)

    def __init__(self, value: AlgebraistLayoutPathLike) -> None:
        object.__setattr__(self, "parts", self._normalize(value))

    @staticmethod
    def _normalize(value: AlgebraistLayoutPathLike) -> tuple[str, ...]:
        if isinstance(value, str):
            parts = tuple(value.split("."))
        else:
            parts = tuple(value)

        if not parts:
            raise ValueError("AlgebraistLayoutPath cannot be empty.")
        for part in parts:
            if not isinstance(part, str):
                raise TypeError("AlgebraistLayoutPath parts must be strings.")
            if not part:
                raise ValueError("AlgebraistLayoutPath parts cannot be empty.")
            if not part.isidentifier():
                raise ValueError(
                    f"AlgebraistLayoutPath part {part!r} is not a valid identifier."
                )
        return parts

    @property
    def name(self) -> str:
        return "_".join(self.parts)

    def expression(self, root: str) -> str:
        if not root.isidentifier():
            raise ValueError(f"root {root!r} is not a valid identifier.")
        return ".".join((root, *self.parts))

    def __iter__(self):
        return iter(self.parts)

    def __len__(self) -> int:
        return len(self.parts)

    def __str__(self) -> str:
        return ".".join(self.parts)
