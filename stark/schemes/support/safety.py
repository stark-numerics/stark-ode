"""Scheme-facing safety protocol."""

from __future__ import annotations

from typing import Protocol


class SchemeSafety(Protocol):
    """Safety switches consumed by scheme orchestration."""

    apply_delta: bool


__all__ = ["SchemeSafety"]
