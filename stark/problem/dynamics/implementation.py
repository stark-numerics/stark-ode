"""Scheme-facing dynamics implementations.

Objects in this module are produced by recognised dynamics signatures and
called by schemes through the common `implementation(interval, state, out)`
contract. They adapt user-friendly call shapes to the mutable translation
objects used internally by STARK.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from stark.core.contracts.accelerator import Accelerator
from stark.problem.dynamics.returns import (
    assign_returned_fields,
    assign_returned_translation,
)


class DynamicsImplementation(Protocol):
    """Scheme-facing dynamics callable prepared from a user declaration."""

    def __call__(self, interval: Any, state: Any, out: Any) -> None: ...

    def accelerate(self, accelerator: Accelerator) -> DynamicsImplementation: ...


@dataclass(frozen=True, slots=True)
class DynamicsAdapterAcceptsInstantWrites:
    """Scheme dynamics adapter for `function(t, state, out)`."""

    function: Callable[..., Any]

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.function(interval.present, state, out)

    def accelerate(self, accelerator: Accelerator) -> DynamicsAdapterAcceptsInstantWrites:
        del accelerator
        return self


@dataclass(frozen=True, slots=True)
class DynamicsAdapterAcceptsIntervalWrites:
    """Scheme dynamics adapter for `function(interval, state, out)`."""

    function: Callable[..., Any]

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.function(interval, state, out)

    def accelerate(self, accelerator: Accelerator) -> DynamicsAdapterAcceptsIntervalWrites:
        del accelerator
        return self


@dataclass(frozen=True, slots=True)
class DynamicsAdapterAcceptsInstantReturns:
    """Scheme dynamics adapter for `function(t, state) -> translation`."""

    function: Callable[..., Any]

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        assign_returned_translation(self.function(interval.present, state), out)

    def accelerate(self, accelerator: Accelerator) -> DynamicsAdapterAcceptsInstantReturns:
        del accelerator
        return self


@dataclass(frozen=True, slots=True)
class DynamicsAdapterAcceptsIntervalReturns:
    """Scheme dynamics adapter for `function(interval, state) -> translation`."""

    function: Callable[..., Any]

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        assign_returned_translation(self.function(interval, state), out)

    def accelerate(self, accelerator: Accelerator) -> DynamicsAdapterAcceptsIntervalReturns:
        del accelerator
        return self


@dataclass(frozen=True, slots=True)
class DynamicsKernelAcceptsInstantWrites:
    """Scheme adapter for `kernel(t, *state_fields, *out_fields, *parameters)`."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    translation: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.function(
            interval.present,
            *(getattr(state, name) for name in self.state),
            *(getattr(out, name) for name in self.translation),
            *self.parameters,
        )

    def accelerate(self, accelerator: Accelerator) -> DynamicsKernelAcceptsInstantWrites:
        return DynamicsKernelAcceptsInstantWrites(
            function=accelerator.compile(self.function),
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class DynamicsKernelAcceptsIntervalWrites:
    """Scheme adapter for `kernel(interval, *state_fields, *out_fields, *parameters)`."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    translation: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.function(
            interval,
            *(getattr(state, name) for name in self.state),
            *(getattr(out, name) for name in self.translation),
            *self.parameters,
        )

    def accelerate(self, accelerator: Accelerator) -> DynamicsKernelAcceptsIntervalWrites:
        return DynamicsKernelAcceptsIntervalWrites(
            function=accelerator.compile(self.function),
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class DynamicsKernelAcceptsInstantReturns:
    """Scheme adapter for `kernel(t, *state_fields, *parameters) -> fields`."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    translation: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        result = self.function(
            interval.present,
            *(getattr(state, name) for name in self.state),
            *self.parameters,
        )
        assign_returned_fields(result, out, self.translation)

    def accelerate(self, accelerator: Accelerator) -> DynamicsKernelAcceptsInstantReturns:
        return DynamicsKernelAcceptsInstantReturns(
            function=accelerator.compile(self.function),
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class DynamicsKernelAcceptsIntervalReturns:
    """Scheme adapter for `kernel(interval, *state_fields, *parameters) -> fields`."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    translation: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        result = self.function(
            interval,
            *(getattr(state, name) for name in self.state),
            *self.parameters,
        )
        assign_returned_fields(result, out, self.translation)

    def accelerate(self, accelerator: Accelerator) -> DynamicsKernelAcceptsIntervalReturns:
        return DynamicsKernelAcceptsIntervalReturns(
            function=accelerator.compile(self.function),
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


__all__ = [
    "DynamicsAdapterAcceptsInstantWrites",
    "DynamicsAdapterAcceptsIntervalWrites",
    "DynamicsAdapterAcceptsInstantReturns",
    "DynamicsAdapterAcceptsIntervalReturns",
    "DynamicsImplementation",
    "DynamicsKernelAcceptsInstantWrites",
    "DynamicsKernelAcceptsIntervalWrites",
    "DynamicsKernelAcceptsInstantReturns",
    "DynamicsKernelAcceptsIntervalReturns",
]
