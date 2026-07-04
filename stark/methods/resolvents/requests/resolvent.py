from __future__ import annotations

from typing import Protocol

from stark.core.contracts import (
    BlockLike,
    DynamicsLike,
    IntervalLike,
    StateType,
    TranslationType,
)


class ResolventRequest(Protocol[StateType, TranslationType]):
    """One-item implicit equation request.

    The residual is understood as:

        F(delta) = delta - rhs - alpha * f(interval, origin + delta)

    The request is generic in the state and translation selected by the scheme
    that created it. Resolvents should preserve those types rather than falling
    back to broad package contracts, otherwise IDEs cannot distinguish a valid
    scalar/vector request from an accidental mix of incompatible state shapes.
    """

    @property
    def dynamics(self) -> DynamicsLike[StateType, TranslationType]:
        ...

    @property
    def interval(self) -> IntervalLike:
        ...

    @property
    def origin(self) -> StateType:
        ...

    @property
    def rhs(self) -> BlockLike[TranslationType] | None:
        ...

    @property
    def alpha(self) -> float:
        ...


class ResolventRequestCoupled(Protocol[StateType, TranslationType]):
    """Coupled implicit equation request.

    Coupled requests add the stage coupling data needed by multi-stage
    implicit and IMEX schemes while retaining the same state and translation
    parameters as the underlying dynamics.
    """

    @property
    def dynamics(self) -> DynamicsLike[StateType, TranslationType]:
        ...

    @property
    def interval(self) -> IntervalLike:
        ...

    @property
    def origin(self) -> StateType:
        ...

    @property
    def rhs(self) -> BlockLike[TranslationType] | None:
        ...

    @property
    def step(self) -> float:
        ...

    @property
    def stage_shifts(self) -> tuple[float, ...]:
        ...

    @property
    def matrix(self) -> tuple[tuple[float, ...], ...]:
        ...


__all__ = ["ResolventRequestCoupled", "ResolventRequest"]
