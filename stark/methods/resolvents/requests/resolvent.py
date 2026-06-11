from __future__ import annotations

from typing import Protocol

from stark.block import Block
from stark.contracts import DerivativeLike, IntervalLike, State, Translation


class ResolventRequest(Protocol):
    """One-item implicit equation request.

    The residual is understood as:

        F(delta) = delta - rhs - alpha * f(interval, origin + delta)
    """

    derivative: DerivativeLike
    interval: IntervalLike
    origin: State
    rhs: Block[Translation] | None
    alpha: float


class ResolventRequestCoupled(Protocol):
    """Coupled implicit equation request."""

    derivative: DerivativeLike
    interval: IntervalLike
    origin: State
    rhs: Block[Translation] | None
    step: float
    stage_shifts: tuple[float, ...]
    matrix: tuple[tuple[float, ...], ...]


__all__ = ["ResolventRequestCoupled", "ResolventRequest"]
