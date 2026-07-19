"""Resolvent-facing linearizer implementations.

Objects in this module are produced by recognised linearizer signatures and
called through the common ``linearizer(interval, state, out)`` contract. They
adapt user-friendly linearizer declarations to the mutable operator container
used by implicit resolvents.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from stark.core.contracts.engines.accelerator import Accelerator
from stark.problem.dynamics.returns import assign_returned_fields


class LinearizerImplementation(Protocol):
    """Resolvent-facing linearizer callable prepared from a user declaration."""

    def __call__(self, interval: Any, state: Any, out: Any) -> None: ...

    def accelerate(self, accelerator: Accelerator) -> LinearizerImplementation: ...


@dataclass(frozen=True, slots=True)
class LinearizerAdapterAcceptsInstant:
    """Resolvent linearizer adapter for ``function(t, state, out)``."""

    function: Callable[..., Any]

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.function(interval.present, state, out)

    def accelerate(self, accelerator: Accelerator) -> LinearizerAdapterAcceptsInstant:
        del accelerator
        return self


@dataclass(frozen=True, slots=True)
class LinearizerAdapterAcceptsInterval:
    """Resolvent linearizer adapter for ``function(interval, state, out)``."""

    function: Callable[..., Any]

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.function(interval, state, out)

    def accelerate(self, accelerator: Accelerator) -> LinearizerAdapterAcceptsInterval:
        del accelerator
        return self


@dataclass(frozen=True, slots=True)
class LinearizerKernel:
    """Prepared field-level Jacobian-action kernel that writes target fields."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    source: tuple[str, ...]
    target: tuple[str, ...]
    parameters: tuple[Any, ...] = ()
    accepts_interval: bool = False

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        state_values = tuple(getattr(state, name) for name in self.state)
        function = self.function
        source_names = self.source
        target_names = self.target
        parameters = self.parameters
        instant_or_interval = interval if self.accepts_interval else interval.present

        def apply(source: Any, result: Any) -> None:
            function(
                instant_or_interval,
                *state_values,
                *(getattr(source, name) for name in source_names),
                *(getattr(result, name) for name in target_names),
                *parameters,
            )

        out.apply = apply

    def accelerate(self, accelerator: Accelerator) -> LinearizerKernel:
        return LinearizerKernel(
            function=accelerator.compile(self.function, label="linearizer-apply", cache=True),
            state=self.state,
            source=self.source,
            target=self.target,
            parameters=self.parameters,
            accepts_interval=self.accepts_interval,
        )


@dataclass(frozen=True, slots=True)
class LinearizerKernelReturning:
    """Prepared field-level Jacobian-action kernel that returns target fields."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    source: tuple[str, ...]
    target: tuple[str, ...]
    parameters: tuple[Any, ...] = ()
    accepts_interval: bool = False

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        state_values = tuple(getattr(state, name) for name in self.state)
        function = self.function
        source_names = self.source
        target_names = self.target
        parameters = self.parameters
        instant_or_interval = interval if self.accepts_interval else interval.present

        def apply(source: Any, result: Any) -> None:
            values = function(
                instant_or_interval,
                *state_values,
                *(getattr(source, name) for name in source_names),
                *parameters,
            )
            assign_returned_fields(values, result, target_names)

        out.apply = apply

    def accelerate(self, accelerator: Accelerator) -> LinearizerKernelReturning:
        return LinearizerKernelReturning(
            function=accelerator.compile(self.function, label="linearizer-apply-returning", cache=True),
            state=self.state,
            source=self.source,
            target=self.target,
            parameters=self.parameters,
            accepts_interval=self.accepts_interval,
        )


@dataclass(frozen=True, slots=True)
class LinearizerDense:
    """Prepared field-level dense Jacobian fill kernel."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        del interval
        state_values = tuple(getattr(state, name) for name in self.state)
        function = self.function
        parameters = self.parameters

        def dense_fill(_basis: Any, matrix: Any, row_offset: int, column_offset: int, stride: int) -> None:
            function(
                *state_values,
                matrix,
                row_offset,
                column_offset,
                stride,
                *parameters,
            )

        out.dense_fill = dense_fill

    def accelerate(self, accelerator: Accelerator) -> LinearizerDense:
        return LinearizerDense(
            function=accelerator.compile(self.function, label="linearizer-dense-fill", cache=True),
            state=self.state,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class LinearizerOperator:
    """Prepared combined apply/dense linearizer."""

    apply: LinearizerImplementation | None = None
    dense: LinearizerImplementation | None = None

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        if self.apply is not None:
            self.apply(interval, state, out)
        if self.dense is not None:
            self.dense(interval, state, out)

    def accelerate(self, accelerator: Accelerator) -> LinearizerOperator:
        return LinearizerOperator(
            apply=self.apply.accelerate(accelerator) if self.apply is not None else None,
            dense=self.dense.accelerate(accelerator) if self.dense is not None else None,
        )


__all__ = [
    "LinearizerAdapterAcceptsInstant",
    "LinearizerAdapterAcceptsInterval",
    "LinearizerDense",
    "LinearizerImplementation",
    "LinearizerKernel",
    "LinearizerKernelReturning",
    "LinearizerOperator",
]
