from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ResolventDynamics:
    """Stable callable wrapper for a dynamics used inside a resolvent."""

    raw: Any

    def __call__(self, interval, state, out) -> Any:
        return self.raw(interval, state, out)


@dataclass(slots=True)
class ResolventLinearizer:
    """Stable callable wrapper for a linearizer used inside a resolvent."""

    raw: Any

    def __call__(self, interval, state, out) -> Any:
        return self.raw(interval, state, out)


__all__ = ["ResolventDynamics", "ResolventLinearizer"]
