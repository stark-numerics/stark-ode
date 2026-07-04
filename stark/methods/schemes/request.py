from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from stark.core.contracts import (
    BlockLike,
    DerivativeLike,
    IntervalLike,
    StateType,
    TranslationType,
)


@dataclass(frozen=True, slots=True)
class SchemeResolventRequest(Generic[StateType, TranslationType]):
    """Concrete one-item implicit equation request supplied by a scheme.

    The request carries the state and translation types selected by the owning
    scheme. Keeping those types intact lets resolvents and tests see that a
    scalar-state request, for example, contains scalar translation blocks rather
    than anonymous package-wide `Translation` values.
    """

    derivative: DerivativeLike[StateType, TranslationType]
    interval: IntervalLike
    origin: StateType
    rhs: BlockLike[TranslationType] | None
    alpha: float


@dataclass(frozen=True, slots=True)
class SchemeResolventRequestCoupled(Generic[StateType, TranslationType]):
    """Concrete coupled implicit equation request supplied by a scheme.

    Coupled schemes solve several stage equations as one block problem. This
    request preserves the same concrete state and translation types as the
    uncoupled request while adding the stage-shift and tableau matrix data
    needed to build the coupled residual.
    """

    derivative: DerivativeLike[StateType, TranslationType]
    interval: IntervalLike
    origin: StateType
    rhs: BlockLike[TranslationType] | None
    step: float
    stage_shifts: tuple[float, ...]
    matrix: tuple[tuple[float, ...], ...]


__all__ = ["SchemeResolventRequestCoupled", "SchemeResolventRequest"]
