from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InverterDescriptor:
    """Display metadata for an inverter."""

    short_name: str
    full_name: str


__all__ = ["InverterDescriptor"]
