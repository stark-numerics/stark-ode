"""Scheme-facing derivative implementations.

Objects in this module are produced by recognised derivative signatures and
called by schemes through the common `implementation(interval, state, out)`
contract. They adapt user-friendly call shapes to the mutable translation
objects used internally by STARK.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from stark.core.contracts.accelerator import Accelerator
from stark.problem.derivative.returns import (
    assign_returned_fields,
    assign_returned_translation,
)


class DerivativeImplementation(Protocol):
    """Scheme-facing derivative callable prepared from a user declaration."""

    def __call__(self, interval: Any, state: Any, out: Any) -> None: ...

    def accelerate(self, accelerator: Accelerator) -> DerivativeImplementation: ...


@dataclass(frozen=True, slots=True)
class DerivativeAdapterAcceptsInstantWrites:
    """Scheme derivative adapter for `function(t, state, out)`."""

    function: Callable[..., Any]

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.function(interval.present, state, out)

    def accelerate(self, accelerator: Accelerator) -> DerivativeAdapterAcceptsInstantWrites:
        del accelerator
        return self


@dataclass(frozen=True, slots=True)
class DerivativeAdapterAcceptsIntervalWrites:
    """Scheme derivative adapter for `function(interval, state, out)`."""

    function: Callable[..., Any]

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.function(interval, state, out)

    def accelerate(self, accelerator: Accelerator) -> DerivativeAdapterAcceptsIntervalWrites:
        del accelerator
        return self


@dataclass(frozen=True, slots=True)
class DerivativeAdapterAcceptsInstantReturns:
    """Scheme derivative adapter for `function(t, state) -> translation`."""

    function: Callable[..., Any]

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        assign_returned_translation(self.function(interval.present, state), out)

    def accelerate(self, accelerator: Accelerator) -> DerivativeAdapterAcceptsInstantReturns:
        del accelerator
        return self


@dataclass(frozen=True, slots=True)
class DerivativeAdapterAcceptsIntervalReturns:
    """Scheme derivative adapter for `function(interval, state) -> translation`."""

    function: Callable[..., Any]

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        assign_returned_translation(self.function(interval, state), out)

    def accelerate(self, accelerator: Accelerator) -> DerivativeAdapterAcceptsIntervalReturns:
        del accelerator
        return self


@dataclass(frozen=True, slots=True)
class DerivativeKernelAcceptsInstantWrites:
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

    def accelerate(self, accelerator: Accelerator) -> DerivativeKernelAcceptsInstantWrites:
        return DerivativeKernelAcceptsInstantWrites(
            function=accelerator.compile(self.function),
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class DerivativeKernelAcceptsIntervalWrites:
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

    def accelerate(self, accelerator: Accelerator) -> DerivativeKernelAcceptsIntervalWrites:
        return DerivativeKernelAcceptsIntervalWrites(
            function=accelerator.compile(self.function),
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class DerivativeKernelAcceptsInstantReturns:
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

    def accelerate(self, accelerator: Accelerator) -> DerivativeKernelAcceptsInstantReturns:
        return DerivativeKernelAcceptsInstantReturns(
            function=accelerator.compile(self.function),
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class DerivativeKernelAcceptsIntervalReturns:
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

    def accelerate(self, accelerator: Accelerator) -> DerivativeKernelAcceptsIntervalReturns:
        return DerivativeKernelAcceptsIntervalReturns(
            function=accelerator.compile(self.function),
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


__all__ = [
    "DerivativeAdapterAcceptsInstantWrites",
    "DerivativeAdapterAcceptsIntervalWrites",
    "DerivativeAdapterAcceptsInstantReturns",
    "DerivativeAdapterAcceptsIntervalReturns",
    "DerivativeImplementation",
    "DerivativeKernelAcceptsInstantWrites",
    "DerivativeKernelAcceptsIntervalWrites",
    "DerivativeKernelAcceptsInstantReturns",
    "DerivativeKernelAcceptsIntervalReturns",
]
