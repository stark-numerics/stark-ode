"""User-facing derivative declarations.

Signature objects record the shape of a user derivative and any metadata needed
to turn it into scheme-facing code. In this module, "signature" means the
recognised STARK declaration shape, not Python's raw `inspect.signature(...)`.

Kernel declarations always receive either the current time value or the full
interval object. There is intentionally no autonomous kernel shape: even when a
kernel does not use time, accepting and discarding it keeps the derivative
contract explicit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, overload, runtime_checkable

from stark.core.contracts.derivative import DerivativeLike
from stark.problem.derivative.implementation import (
    DerivativeAdapterAcceptsInstantWrites,
    DerivativeAdapterAcceptsIntervalWrites,
    DerivativeAdapterAcceptsInstantReturns,
    DerivativeAdapterAcceptsIntervalReturns,
    DerivativeImplementation,
    DerivativeKernelAcceptsInstantWrites,
    DerivativeKernelAcceptsIntervalWrites,
    DerivativeKernelAcceptsInstantReturns,
    DerivativeKernelAcceptsIntervalReturns,
)
from stark.problem.derivative.split import DerivativeSplit


@runtime_checkable
class DerivativeSignature(Protocol):
    """Recognised user derivative declaration that can produce scheme code."""

    def implementation(self) -> DerivativeImplementation: ...


class DerivativeStyle:
    """Construct recognised derivative signatures from user callables.

    These names are intentionally user-facing. `accepts_instant_*` variants
    receive the current time value, `accepts_interval_*` variants receive the
    whole interval object, and `*_returns` names distinguish functions that
    return translation values from functions that write into an output object.
    """

    @staticmethod
    def accepts_instant_returns(function: Callable[..., Any]) -> DerivativeSignature:
        """Recognise `function(t, state) -> translation` as a pure derivative."""

        return DerivativeSignatureAcceptsInstantReturns(function)

    @staticmethod
    def accepts_interval_returns(function: Callable[..., Any]) -> DerivativeSignature:
        """Recognise `function(interval, state) -> translation` as a pure derivative."""

        return DerivativeSignatureAcceptsIntervalReturns(function)

    @staticmethod
    def accepts_instant_writes(function: Callable[..., Any]) -> DerivativeSignature:
        """Recognise `function(t, state, out)` as an in-place derivative."""

        return DerivativeSignatureAcceptsInstantWrites(function)

    @staticmethod
    def accepts_interval_writes(function: Callable[..., Any]) -> DerivativeSignature:
        """Recognise `function(interval, state, out)` as an in-place derivative."""

        return DerivativeSignatureAcceptsIntervalWrites(function)

    @staticmethod
    def split(*, implicit: DerivativeLike, explicit: DerivativeLike) -> DerivativeSplit:
        """Recognise an implicit-explicit derivative split."""

        return DerivativeSplit(
            implicit=implicit,
            explicit=explicit,
        )

    @staticmethod
    @overload
    def kernel_accepts_instant_writes(
        function: None = None,
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> Callable[[Callable[..., Any]], DerivativeSignatureKernelAcceptsInstantWrites]:
        ...

    @staticmethod
    @overload
    def kernel_accepts_instant_writes(
        function: Callable[..., Any],
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> DerivativeSignatureKernelAcceptsInstantWrites:
        ...

    @staticmethod
    def kernel_accepts_instant_writes(
        function: Callable[..., Any] | None = None,
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> (
        DerivativeSignatureKernelAcceptsInstantWrites
        | Callable[[Callable[..., Any]], DerivativeSignatureKernelAcceptsInstantWrites]
    ):
        """Recognise `kernel(t, *state_fields, *out_fields, *parameters)`.

        Kernel styles are for field-level array code where unpacking a full
        STARK state object would add noise or block backend acceleration. This
        writing variant receives mutable translation fields and fills them in
        place, which is the natural shape for NumPy, CuPy, and Numba kernels.
        """

        def recognise(target: Callable[..., Any]) -> DerivativeSignatureKernelAcceptsInstantWrites:
            return DerivativeSignatureKernelAcceptsInstantWrites(
                function=target,
                state=state,
                translation=translation,
                parameters=tuple(parameters),
            )

        if function is None:
            return recognise
        return recognise(function)

    @staticmethod
    @overload
    def kernel_accepts_interval_writes(
        function: None = None,
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> Callable[[Callable[..., Any]], DerivativeSignatureKernelAcceptsIntervalWrites]:
        ...

    @staticmethod
    @overload
    def kernel_accepts_interval_writes(
        function: Callable[..., Any],
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> DerivativeSignatureKernelAcceptsIntervalWrites:
        ...

    @staticmethod
    def kernel_accepts_interval_writes(
        function: Callable[..., Any] | None = None,
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> (
        DerivativeSignatureKernelAcceptsIntervalWrites
        | Callable[[Callable[..., Any]], DerivativeSignatureKernelAcceptsIntervalWrites]
    ):
        """Recognise `kernel(interval, *state_fields, *out_fields, *parameters)`.

        This is the interval-aware writing kernel form. Use it when the
        derivative needs more than the current time value, such as the active
        step size or interval endpoint metadata.
        """

        def recognise(target: Callable[..., Any]) -> DerivativeSignatureKernelAcceptsIntervalWrites:
            return DerivativeSignatureKernelAcceptsIntervalWrites(
                function=target,
                state=state,
                translation=translation,
                parameters=tuple(parameters),
            )

        if function is None:
            return recognise
        return recognise(function)

    @staticmethod
    @overload
    def kernel_accepts_instant_returns(
        function: None = None,
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> Callable[[Callable[..., Any]], DerivativeSignatureKernelAcceptsInstantReturns]:
        ...

    @staticmethod
    @overload
    def kernel_accepts_instant_returns(
        function: Callable[..., Any],
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> DerivativeSignatureKernelAcceptsInstantReturns:
        ...

    @staticmethod
    def kernel_accepts_instant_returns(
        function: Callable[..., Any] | None = None,
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> (
        DerivativeSignatureKernelAcceptsInstantReturns
        | Callable[[Callable[..., Any]], DerivativeSignatureKernelAcceptsInstantReturns]
    ):
        """Recognise `kernel(t, *state_fields, *parameters) -> fields`.

        Returning kernels are useful for backends with immutable array values,
        notably JAX. The kernel returns field values and STARK copies them into
        the scheme translation object.
        """

        def recognise(target: Callable[..., Any]) -> DerivativeSignatureKernelAcceptsInstantReturns:
            return DerivativeSignatureKernelAcceptsInstantReturns(
                function=target,
                state=state,
                translation=translation,
                parameters=tuple(parameters),
            )

        if function is None:
            return recognise
        return recognise(function)

    @staticmethod
    @overload
    def kernel_accepts_interval_returns(
        function: None = None,
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> Callable[[Callable[..., Any]], DerivativeSignatureKernelAcceptsIntervalReturns]:
        ...

    @staticmethod
    @overload
    def kernel_accepts_interval_returns(
        function: Callable[..., Any],
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> DerivativeSignatureKernelAcceptsIntervalReturns:
        ...

    @staticmethod
    def kernel_accepts_interval_returns(
        function: Callable[..., Any] | None = None,
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> (
        DerivativeSignatureKernelAcceptsIntervalReturns
        | Callable[[Callable[..., Any]], DerivativeSignatureKernelAcceptsIntervalReturns]
    ):
        """Recognise `kernel(interval, *state_fields, *parameters) -> fields`.

        This is the interval-aware returning kernel form for immutable-array
        backends or pure expression code that also needs interval metadata.
        """

        def recognise(target: Callable[..., Any]) -> DerivativeSignatureKernelAcceptsIntervalReturns:
            return DerivativeSignatureKernelAcceptsIntervalReturns(
                function=target,
                state=state,
                translation=translation,
                parameters=tuple(parameters),
            )

        if function is None:
            return recognise
        return recognise(function)


@dataclass(frozen=True, slots=True)
class DerivativeSignatureAcceptsInstantReturns:
    """User derivative with call shape `function(t, state) -> translation`."""

    function: Callable[..., Any]

    def implementation(self) -> DerivativeImplementation:
        return DerivativeAdapterAcceptsInstantReturns(self.function)


@dataclass(frozen=True, slots=True)
class DerivativeSignatureAcceptsIntervalReturns:
    """User derivative with call shape `function(interval, state) -> translation`."""

    function: Callable[..., Any]

    def implementation(self) -> DerivativeImplementation:
        return DerivativeAdapterAcceptsIntervalReturns(self.function)


@dataclass(frozen=True, slots=True)
class DerivativeSignatureAcceptsInstantWrites:
    """User derivative with call shape `function(t, state, out)`."""

    function: Callable[..., Any]

    def implementation(self) -> DerivativeImplementation:
        return DerivativeAdapterAcceptsInstantWrites(self.function)


@dataclass(frozen=True, slots=True)
class DerivativeSignatureAcceptsIntervalWrites:
    """User derivative with call shape `function(interval, state, out)`."""

    function: Callable[..., Any]

    def implementation(self) -> DerivativeImplementation:
        return DerivativeAdapterAcceptsIntervalWrites(self.function)


@dataclass(frozen=True, slots=True)
class DerivativeSignatureKernelAcceptsInstantWrites:
    """Field-level kernel with call shape `kernel(t, *state, *out, *parameters)`."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    translation: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def with_parameters(self, *parameters: Any) -> DerivativeSignatureKernelAcceptsInstantWrites:
        """Return this recognised kernel with problem-specific parameters."""

        return DerivativeSignatureKernelAcceptsInstantWrites(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=tuple(parameters),
        )

    def implementation(self) -> DerivativeImplementation:
        return DerivativeKernelAcceptsInstantWrites(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class DerivativeSignatureKernelAcceptsIntervalWrites:
    """Field-level kernel with call shape `kernel(interval, *state, *out, *parameters)`."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    translation: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def with_parameters(self, *parameters: Any) -> DerivativeSignatureKernelAcceptsIntervalWrites:
        """Return this recognised kernel with problem-specific parameters."""

        return DerivativeSignatureKernelAcceptsIntervalWrites(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=tuple(parameters),
        )

    def implementation(self) -> DerivativeImplementation:
        return DerivativeKernelAcceptsIntervalWrites(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class DerivativeSignatureKernelAcceptsInstantReturns:
    """Field-level kernel with call shape `kernel(t, *state, *parameters) -> fields`."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    translation: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def with_parameters(self, *parameters: Any) -> DerivativeSignatureKernelAcceptsInstantReturns:
        """Return this recognised kernel with problem-specific parameters."""

        return DerivativeSignatureKernelAcceptsInstantReturns(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=tuple(parameters),
        )

    def implementation(self) -> DerivativeImplementation:
        return DerivativeKernelAcceptsInstantReturns(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class DerivativeSignatureKernelAcceptsIntervalReturns:
    """Field-level kernel with call shape `kernel(interval, *state, *parameters) -> fields`."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    translation: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def with_parameters(self, *parameters: Any) -> DerivativeSignatureKernelAcceptsIntervalReturns:
        """Return this recognised kernel with problem-specific parameters."""

        return DerivativeSignatureKernelAcceptsIntervalReturns(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=tuple(parameters),
        )

    def implementation(self) -> DerivativeImplementation:
        return DerivativeKernelAcceptsIntervalReturns(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


__all__ = [
    "DerivativeSignature",
    "DerivativeSignatureAcceptsInstantWrites",
    "DerivativeSignatureAcceptsIntervalWrites",
    "DerivativeSignatureAcceptsInstantReturns",
    "DerivativeSignatureAcceptsIntervalReturns",
    "DerivativeSignatureKernelAcceptsInstantWrites",
    "DerivativeSignatureKernelAcceptsIntervalWrites",
    "DerivativeSignatureKernelAcceptsInstantReturns",
    "DerivativeSignatureKernelAcceptsIntervalReturns",
    "DerivativeStyle",
]
