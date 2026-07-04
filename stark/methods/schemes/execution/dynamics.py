from __future__ import annotations

from typing import Protocol, Any, runtime_checkable



@runtime_checkable
class SchemeDynamics(Protocol):
    """Callable dynamics surface accepted by built-in schemes."""

    def __call__(self, interval, state, out) -> Any: ...


__all__ = ["SchemeDynamics"]
