"""Canonical derivative wrapper used by `System`.

`Derivative` is the problem-layer object that accepts either a plain Python
callable or a recognised derivative signature and exposes the scheme-facing
`derivative(interval, state, out)` contract.

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
from stark.core.contracts.derivative import DerivativeLike
from stark.problem.derivative.implementation import DerivativeImplementation
from stark.problem.derivative.signature import DerivativeSignature, DerivativeStyle
from stark.problem.derivative.split import DerivativeSplit


@dataclass(frozen=True, slots=True)
class Derivative:
    """Canonical derivative declaration accepted by `System`.

    Plain callables are inferred as either `function(t, state) -> translation`
    or `function(t, state, out) -> None`. Other shapes should be declared with
    `Derivative.accepts_instant_returns`,
    `Derivative.accepts_instant_writes`, the `accepts_interval_*` variants, or
    the lower-level `DerivativeStyle.kernel_*` helpers.
    """

    source: Callable[..., Any] | DerivativeSignature
    implementation: DerivativeImplementation = field(init=False, repr=False)

    def __post_init__(self) -> None:
        source = self.source
        if isinstance(source, DerivativeSignature):
            implementation = source.implementation()
        elif callable(source):
            implementation = self.from_callable(source).implementation()
        else:
            raise TypeError("Derivative requires a callable or derivative signature.")

        object.__setattr__(self, "implementation", implementation)

    @classmethod
    def from_callable(cls, function: Callable[..., Any]) -> DerivativeSignature:
        """Infer common plain derivative signatures.

        Two required positional arguments mean return style:
        `function(t, state) -> translation`. Three required positional
        arguments mean in-place style: `function(t, state, out)`.
        """

        try:
            inspected = signature(function)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Plain derivative callables must expose a visible signature. "
                "Use DerivativeStyle when inference is not possible."
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
            return DerivativeStyle.accepts_instant_returns(function)
        if len(positional) == 3:
            return DerivativeStyle.accepts_instant_writes(function)

        raise TypeError(
            "Plain derivative callables must use function(t, state) or "
            "function(t, state, out). Use DerivativeStyle.accepts_interval_returns(...) "
            "or DerivativeStyle.accepts_interval_writes(...) for interval-aware derivatives."
        )

    @classmethod
    def accepts_instant_returns(cls, function: Callable[..., Any]) -> DerivativeSignature:
        """Declare a pure derivative `function(t, state) -> translation`."""

        return DerivativeStyle.accepts_instant_returns(function)

    @classmethod
    def accepts_interval_returns(cls, function: Callable[..., Any]) -> DerivativeSignature:
        """Declare a pure derivative `function(interval, state) -> translation`."""

        return DerivativeStyle.accepts_interval_returns(function)

    @classmethod
    def accepts_instant_writes(cls, function: Callable[..., Any]) -> DerivativeSignature:
        """Declare an in-place derivative `function(t, state, out)`."""

        return DerivativeStyle.accepts_instant_writes(function)

    @classmethod
    def accepts_interval_writes(cls, function: Callable[..., Any]) -> DerivativeSignature:
        """Declare an in-place derivative `function(interval, state, out)`."""

        return DerivativeStyle.accepts_interval_writes(function)

    @classmethod
    def split(cls, *, implicit: DerivativeLike, explicit: DerivativeLike) -> DerivativeSplit:
        """Declare an implicit-explicit derivative split."""

        return DerivativeStyle.split(implicit=implicit, explicit=explicit)

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.implementation(interval, state, out)

    def accelerate(self, accelerator: Accelerator) -> DerivativeImplementation:
        return self.implementation.accelerate(accelerator)


__all__ = ["Derivative"]
