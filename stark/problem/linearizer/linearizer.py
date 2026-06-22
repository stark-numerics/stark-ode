"""Canonical linearizer wrapper used by `System`.

`Linearizer` is the problem-layer object that accepts either a plain Python
callable or a recognised linearizer signature and exposes the resolvent-facing
``linearizer(interval, state, out)`` contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import Parameter, signature as inspect_signature
from typing import Any

from stark.core.contracts.accelerator import Accelerator
from stark.problem.linearizer.implementation import LinearizerImplementation
from stark.problem.linearizer.signature import LinearizerSignature, LinearizerStyle


@dataclass(frozen=True, slots=True)
class Linearizer:
    """Canonical linearizer declaration accepted by `System`.

    Plain callables are inferred as interval-aware in-place linearizers:
    ``function(interval, state, out)``. Other shapes should be declared with
    `Linearizer.accepts_instant_writes`, `LinearizerStyle.kernel_*`, or
    `LinearizerStyle.operator`.
    """

    source: Callable[..., Any] | LinearizerSignature
    implementation: LinearizerImplementation = field(init=False, repr=False)

    def __post_init__(self) -> None:
        source = self.source
        if isinstance(source, LinearizerSignature):
            implementation = source.implementation()
        elif callable(source):
            implementation = self.from_callable(source).implementation()
        else:
            raise TypeError("Linearizer requires a callable or linearizer signature.")

        object.__setattr__(self, "implementation", implementation)

    @classmethod
    def from_callable(cls, function: Callable[..., Any]) -> LinearizerSignature:
        """Infer the existing plain linearizer convention.

        Three required positional arguments are treated as the resolvent-facing
        interval form: ``function(interval, state, out)``. Use
        ``LinearizerStyle.accepts_instant_writes`` explicitly for
        ``function(t, state, out)``.
        """

        try:
            inspected = inspect_signature(function)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Plain linearizer callables must expose a visible signature. "
                "Use LinearizerStyle when inference is not possible."
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

        if len(positional) == 3:
            return LinearizerStyle.accepts_interval_writes(function)

        raise TypeError(
            "Plain linearizer callables must use function(interval, state, out). "
            "Use LinearizerStyle for field-level kernels or instant-time callables."
        )

    @classmethod
    def accepts_instant_writes(cls, function: Callable[..., Any]) -> LinearizerSignature:
        """Declare an in-place linearizer `function(t, state, out)`."""

        return LinearizerStyle.accepts_instant_writes(function)

    @classmethod
    def accepts_interval_writes(cls, function: Callable[..., Any]) -> LinearizerSignature:
        """Declare an in-place linearizer `function(interval, state, out)`."""

        return LinearizerStyle.accepts_interval_writes(function)

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.implementation(interval, state, out)

    def accelerate(self, accelerator: Accelerator) -> LinearizerImplementation:
        return self.implementation.accelerate(accelerator)


__all__ = ["Linearizer"]
