from __future__ import annotations

from dataclasses import dataclass

from stark.block import Block
from stark.contracts import Derivative, IntervalLike, State, Translation


@dataclass(frozen=True, slots=True)
class SchemeStageProblem:
    """Concrete one-stage problem supplied by an implicit scheme."""

    derivative: Derivative
    interval: IntervalLike
    origin: State
    rhs: Block[Translation] | None
    alpha: float


@dataclass(frozen=True, slots=True)
class SchemeStageProblemCoupled:
    """Concrete coupled-stage problem supplied by an implicit RK scheme."""

    derivative: Derivative
    interval: IntervalLike
    origin: State
    rhs: Block[Translation] | None
    step: float
    stage_shifts: tuple[float, ...]
    matrix: tuple[tuple[float, ...], ...]


__all__ = ["SchemeStageProblemCoupled", "SchemeStageProblem"]
