"""Canonical dynamics wrapper used by `System`.

`Dynamics` is the problem-layer object that accepts either a plain Python
callable or a recognised dynamics signature and exposes the scheme-facing
`dynamics(interval, state, out)` contract.

This module intentionally keeps only the wrapper and plain-callable inference.
The declaration vocabulary lives in `signature.py`, the scheme-facing adapters
live in `implementation.py`, and return-value assignment policy lives in
`returns.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import Any

from stark.core.contracts.accelerator import Accelerator
from stark.core.contracts.dynamics import DynamicsLike
from stark.problem.dynamics.implementation import DynamicsImplementation
from stark.problem.dynamics.signature import DynamicsSignature, DynamicsStyle
from stark.problem.dynamics.split import DynamicsSplit


@dataclass(frozen=True, slots=True)
class Dynamics:
    """Canonical dynamics declaration accepted by `System`.

    Plain callables are inferred as either `function(t, state) -> translation`
    or `function(t, state, out) -> None`. Other shapes should be declared with
    `Dynamics.accepts_instant_returns`,
    `Dynamics.accepts_instant_writes`, the `accepts_interval_*` variants, or
    the lower-level `DynamicsStyle.kernel_*` helpers.
    """

    source: Callable[..., Any] | DynamicsSignature
    implementation: DynamicsImplementation = field(init=False, repr=False)

    def __post_init__(self) -> None:
        source = self.source
        if isinstance(source, DynamicsSignature):
            implementation = source.implementation()
        elif callable(source):
            implementation = self.from_callable(source).implementation()
        else:
            raise TypeError("Dynamics requires a callable or dynamics signature.")

        object.__setattr__(self, "implementation", implementation)

    @classmethod
    def from_callable(cls, function: Callable[..., Any]) -> DynamicsSignature:
        """Infer common plain dynamics signatures.

        Two required positional arguments mean return style:
        `function(t, state) -> translation`. Three required positional
        arguments mean in-place style: `function(t, state, out)`.
        """

        try:
            inspected = signature(function)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Plain dynamics callables must expose a visible signature. "
                "Use DynamicsStyle when inference is not possible."
            ) from exc

        positional = [
            parameter
            for parameter in inspected.parameters.values()
            if parameter.kind
            in (
                Parameter.POSITIONAL_ONLY,
                Parameter.POSITIONAL_OR_KEYWORD,
            )
            and parameter.default is Parameter.empty
        ]

        if len(positional) == 2:
            return DynamicsStyle.accepts_instant_returns(function)
        if len(positional) == 3:
            return DynamicsStyle.accepts_instant_writes(function)

        raise TypeError(
            "Plain dynamics callables must use function(t, state) or "
            "function(t, state, out). Use DynamicsStyle.accepts_interval_returns(...) "
            "or DynamicsStyle.accepts_interval_writes(...) for interval-aware dynamics."
        )

    @classmethod
    def accepts_instant_returns(cls, function: Callable[..., Any]) -> DynamicsSignature:
        """Declare a pure dynamics `function(t, state) -> translation`."""

        return DynamicsStyle.accepts_instant_returns(function)

    @classmethod
    def accepts_interval_returns(cls, function: Callable[..., Any]) -> DynamicsSignature:
        """Declare a pure dynamics `function(interval, state) -> translation`."""

        return DynamicsStyle.accepts_interval_returns(function)

    @classmethod
    def accepts_instant_writes(cls, function: Callable[..., Any]) -> DynamicsSignature:
        """Declare an in-place dynamics `function(t, state, out)`."""

        return DynamicsStyle.accepts_instant_writes(function)

    @classmethod
    def accepts_interval_writes(cls, function: Callable[..., Any]) -> DynamicsSignature:
        """Declare an in-place dynamics `function(interval, state, out)`."""

        return DynamicsStyle.accepts_interval_writes(function)

    @classmethod
    def split(cls, *, implicit: DynamicsLike, explicit: DynamicsLike) -> DynamicsSplit:
        """Declare an implicit-explicit dynamics split."""

        return DynamicsStyle.split(implicit=implicit, explicit=explicit)

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.implementation(interval, state, out)

    def accelerate(self, accelerator: Accelerator) -> DynamicsImplementation:
        return self.implementation.accelerate(accelerator)


__all__ = ["Dynamics"]
