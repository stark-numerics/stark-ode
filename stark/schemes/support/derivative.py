from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SchemeDerivative:
    """Stable callable wrapper for a scheme derivative."""

    raw: Any

    def __call__(self, interval, state, out) -> Any:
        return self.raw(interval, state, out)


__all__ = ["SchemeDerivative"]
