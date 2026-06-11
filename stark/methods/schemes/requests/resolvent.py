from __future__ import annotations

from dataclasses import dataclass

from stark.block import Block
from stark.contracts import DerivativeLike, IntervalLike, State, Translation


@dataclass(frozen=True, slots=True)
class SchemeResolventRequest:
    """Concrete one-item implicit equation request supplied by a scheme."""

    derivative: DerivativeLike
    interval: IntervalLike
    origin: State
    rhs: Block[Translation] | None
    alpha: float


@dataclass(frozen=True, slots=True)
class SchemeResolventRequestCoupled:
    """Concrete coupled implicit equation request supplied by a scheme."""

    derivative: DerivativeLike
    interval: IntervalLike
    origin: State
    rhs: Block[Translation] | None
    step: float
    stage_shifts: tuple[float, ...]
    matrix: tuple[tuple[float, ...], ...]


__all__ = ["SchemeResolventRequestCoupled", "SchemeResolventRequest"]
