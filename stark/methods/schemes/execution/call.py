from __future__ import annotations

from collections.abc import Callable

from stark.core.contracts import IntervalLike, State

SchemeCall = Callable[[IntervalLike, State], float]

__all__ = ["SchemeCall"]
