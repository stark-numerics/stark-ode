from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResolventDescriptor:
    short_name: str
    full_name: str

    def __repr__(self) -> str:
        return f"ResolventDescriptor(short_name={self.short_name!r}, full_name={self.full_name!r})"

    def __str__(self) -> str:
        return f"{self.short_name} ({self.full_name})"


__all__ = ["ResolventDescriptor"]









