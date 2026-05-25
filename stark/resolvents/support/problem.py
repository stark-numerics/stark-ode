from __future__ import annotations

from typing import Protocol

from stark.block import Block
from stark.contracts import Derivative, IntervalLike, State, Translation


class ResolventStageProblem(Protocol):
    """One-stage implicit residual problem.

    The residual is understood as:

        F(delta) = delta - rhs - alpha * f(interval, origin + delta)
    """

    derivative: Derivative
    interval: IntervalLike
    origin: State
    rhs: Block[Translation] | None
    alpha: float


class ResolventCoupledStageProblem(Protocol):
    """Coupled implicit Runge-Kutta stage residual problem."""

    derivative: Derivative
    interval: IntervalLike
    origin: State
    rhs: Block[Translation] | None
    step: float
    stage_shifts: tuple[float, ...]
    matrix: tuple[tuple[float, ...], ...]


__all__ = ["ResolventCoupledStageProblem", "ResolventStageProblem"]
