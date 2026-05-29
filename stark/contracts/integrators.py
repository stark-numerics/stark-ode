"""Contracts for trajectory-building integrators."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

from stark.contracts.intervals import IntervalLike
from stark.contracts.marchers import MarcherLike
from stark.contracts.states import State


class IntegratorLike(Protocol):
    """
    Protocol for trajectory-building workers built on top of a marcher.

    Integrators repeatedly call a marcher until the interval reaches its stop
    time, yielding either snapshot copies or live mutable objects along the
    way.
    """

    def __call__(
        self,
        marcher: MarcherLike,
        interval: IntervalLike,
        state: State,
        checkpoints: Any | None = None,
    ) -> Iterator[tuple[IntervalLike, State]]:
        ...

    def live(
        self,
        marcher: MarcherLike,
        interval: IntervalLike,
        state: State,
        checkpoints: Any | None = None,
    ) -> Iterator[tuple[IntervalLike, State]]:
        ...


__all__ = ["IntegratorLike"]
