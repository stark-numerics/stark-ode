from __future__ import annotations

from typing import Protocol, Any, runtime_checkable



@runtime_checkable
class SchemeDerivative(Protocol):
    """Callable derivative surface accepted by built-in schemes."""

    def __call__(self, interval, state, out) -> Any: ...


__all__ = ["SchemeDerivative"]
