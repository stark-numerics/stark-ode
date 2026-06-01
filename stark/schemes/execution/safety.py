"""Scheme-facing safety protocol."""

from __future__ import annotations

from typing import Protocol


class SchemeSafety(Protocol):
    """Scheme-facing view of executor safety policy."""

    pass


__all__ = ["SchemeSafety"]
