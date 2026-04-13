from __future__ import annotations

from dataclasses import dataclass

from stark.tolerance import Tolerance


@dataclass(slots=True)
class SchemeTolerance(Tolerance):
    """Scheme-facing tolerance object for adaptive step acceptance."""


__all__ = ["SchemeTolerance"]
