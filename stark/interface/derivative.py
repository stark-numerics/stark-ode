from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import Any, Protocol, overload, runtime_checkable

from stark.contracts.accelerator import Accelerator


class StarkDerivativeImplementation(Protocol):
    """Scheme-facing derivative callable prepared from a user signature."""

    def __call__(self, interval: Any, state: Any, out: Any) -> None: ...

    def accelerate(self, accelerator: Accelerator) -> "StarkDerivativeImplementation": ...


@runtime_checkable
class StarkDerivativeSignature(Protocol):
    """Recognised user derivative signature that can produce a scheme callable."""

    def implementation(self) -> StarkDerivativeImplementation: ...


class StarkDerivativeStyle:
    """Construct recognised derivative signatures from user callables."""

    @staticmethod
    def in_place(function: Callable[..., Any]) -> StarkDerivativeSignature:
        """Recognise `function(t, state, out)` as an in-place derivative."""

        return StarkDerivativeSignatureTimeInPlace(function)

    @staticmethod
    def interval_in_place(function: Callable[..., Any]) -> StarkDerivativeSignature:
        """Recognise `function(interval, state, out)` as an in-place derivative."""

        return StarkDerivativeSignatureIntervalInPlace(function)

    @staticmethod
    @overload
    def kernel(
        function: None = None,
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> Callable[[Callable[..., Any]], "StarkDerivativeSignatureKernel"]:
        ...

    @staticmethod
    @overload
    def kernel(
        function: Callable[..., Any],
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> "StarkDerivativeSignatureKernel":
        ...

    @staticmethod
    def kernel(
        function: Callable[..., Any] | None = None,
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> (
        "StarkDerivativeSignatureKernel"
        " | Callable[[Callable[..., Any]], StarkDerivativeSignatureKernel]"
    ):
        """Recognise a field-level kernel suitable for backend acceleration."""

        def recognise(target: Callable[..., Any]) -> StarkDerivativeSignatureKernel:
            return StarkDerivativeSignatureKernel(
                function=target,
                state=state,
                translation=translation,
                parameters=tuple(parameters),
            )

        if function is None:
            return recognise
        return recognise(function)


@dataclass(frozen=True, slots=True)
class StarkDerivativeSignatureTimeInPlace:
    """User derivative with call shape `function(t, state, out)`."""

    function: Callable[..., Any]

    def implementation(self) -> StarkDerivativeImplementation:
        return StarkDerivativeTimeInPlace(self.function)


@dataclass(frozen=True, slots=True)
class StarkDerivativeSignatureIntervalInPlace:
    """User derivative with call shape `function(interval, state, out)`."""

    function: Callable[..., Any]

    def implementation(self) -> StarkDerivativeImplementation:
        return StarkDerivativeIntervalInPlace(self.function)


@dataclass(frozen=True, slots=True)
class StarkDerivativeSignatureKernel:
    """Field-level derivative kernel with explicit state and translation fields."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    translation: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def with_parameters(self, *parameters: Any) -> "StarkDerivativeSignatureKernel":
        """Return this recognised kernel with problem-specific parameters."""

        return StarkDerivativeSignatureKernel(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=tuple(parameters),
        )

    def implementation(self) -> StarkDerivativeImplementation:
        return StarkDerivativeKernel(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class StarkDerivative:
    """Canonical derivative declaration accepted by `StarkSystem`."""

    source: Callable[..., Any] | StarkDerivativeSignature
    implementation: StarkDerivativeImplementation = field(init=False, repr=False)

    def __post_init__(self) -> None:
        source = self.source
        if isinstance(source, StarkDerivativeSignature):
            implementation = source.implementation()
        elif callable(source):
            implementation = self.from_callable(source).implementation()
        else:
            raise TypeError("StarkDerivative requires a callable or derivative signature.")

        object.__setattr__(self, "implementation", implementation)

    @classmethod
    def from_callable(cls, function: Callable[..., Any]) -> StarkDerivativeSignature:
        """Infer the common `function(t, state, out)` in-place signature."""

        try:
            inspected = signature(function)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Plain derivative callables must expose a visible signature. "
                "Use StarkDerivativeStyle when inference is not possible."
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
            return StarkDerivativeStyle.in_place(function)

        raise TypeError(
            "Plain derivative callables must use function(t, state, out). "
            "Use StarkDerivativeStyle.interval_in_place(...) for interval-aware derivatives."
        )

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.implementation(interval, state, out)

    def accelerate(self, accelerator: Accelerator) -> StarkDerivativeImplementation:
        return self.implementation.accelerate(accelerator)


@dataclass(frozen=True, slots=True)
class StarkDerivativeTimeInPlace:
    """Scheme derivative adapter for `function(t, state, out)`."""

    function: Callable[..., Any]

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.function(interval.present, state, out)

    def accelerate(self, accelerator: Accelerator) -> "StarkDerivativeTimeInPlace":
        del accelerator
        return self


@dataclass(frozen=True, slots=True)
class StarkDerivativeIntervalInPlace:
    """Scheme derivative adapter for `function(interval, state, out)`."""

    function: Callable[..., Any]

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.function(interval, state, out)

    def accelerate(self, accelerator: Accelerator) -> "StarkDerivativeIntervalInPlace":
        del accelerator
        return self


@dataclass(frozen=True, slots=True)
class StarkDerivativeKernel:
    """Scheme derivative adapter for an explicit field-level kernel."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    translation: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        del interval
        self.function(
            *(getattr(state, name) for name in self.state),
            *(getattr(out, name) for name in self.translation),
            *self.parameters,
        )

    def accelerate(self, accelerator: Accelerator) -> "StarkDerivativeKernel":
        return StarkDerivativeKernel(
            function=accelerator.compile(self.function),
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


__all__ = [
    "StarkDerivative",
    "StarkDerivativeImplementation",
    "StarkDerivativeIntervalInPlace",
    "StarkDerivativeKernel",
    "StarkDerivativeSignature",
    "StarkDerivativeSignatureIntervalInPlace",
    "StarkDerivativeSignatureKernel",
    "StarkDerivativeSignatureTimeInPlace",
    "StarkDerivativeStyle",
    "StarkDerivativeTimeInPlace",
]
