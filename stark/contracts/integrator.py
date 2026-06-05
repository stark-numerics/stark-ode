"""Contracts for trajectory-building integrators."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

from stark.contracts.interval import IntervalLike
from stark.contracts.stepper import IntegratorStepperLike
from stark.contracts.state import State


class IntegratorLike(Protocol):
    """
    Protocol for trajectory-building workers built on top of a stepper.

    Integrators repeatedly call a stepper until the interval reaches its stop
    time, yielding either snapshot copies or live mutable objects along the
    way.
    """

    def __call__(
        self,
        stepper: IntegratorStepperLike,
        interval: IntervalLike,
        state: State,
        checkpoints: Any | None = None,
    ) -> Iterator[tuple[IntervalLike, State]]:
        ...

    def live(
        self,
        stepper: IntegratorStepperLike,
        interval: IntervalLike,
        state: State,
        checkpoints: Any | None = None,
    ) -> Iterator[tuple[IntervalLike, State]]:
        ...


__all__ = ["IntegratorLike"]
