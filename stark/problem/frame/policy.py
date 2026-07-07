"""Field policies used by user-facing frame declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class FieldPolicy:
    """Traversal policy for generated frame operations.

    The default ``auto`` policy broadcasts scalar-like fields and emits loops
    for shaped fields. Use explicit policies to request scalar, broadcast,
    looped, or unravelled handling.
    """

    kind: str = "auto"

    known_kinds: ClassVar[frozenset[str]] = frozenset(
        {"auto", "broadcast", "looped", "scalar", "unravel"}
    )

    @classmethod
    def auto(cls) -> "FieldPolicy":
        return cls(kind="auto")

    @classmethod
    def broadcast(cls) -> "FieldPolicy":
        return cls(kind="broadcast")

    @classmethod
    def looped(cls) -> "FieldPolicy":
        return cls(kind="looped")

    @classmethod
    def scalar(cls) -> "FieldPolicy":
        return cls(kind="scalar")

    @classmethod
    def unravel(cls) -> "FieldPolicy":
        return cls(kind="unravel")

    def __post_init__(self) -> None:
        if self.kind not in self.known_kinds:
            raise ValueError(f"Unsupported field policy kind: {self.kind!r}.")


__all__ = ["FieldPolicy"]
