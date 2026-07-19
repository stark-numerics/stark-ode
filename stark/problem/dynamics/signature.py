"""User-facing dynamics declarations.

Signature objects record the shape of a user dynamics and any metadata needed
to turn it into scheme-facing code. In this module, "signature" means the
recognised STARK declaration shape, not Python's raw `inspect.signature(...)`.

Kernel declarations always receive either the current time value or the full
interval object. There is intentionally no autonomous kernel shape: even when a
kernel does not use time, accepting and discarding it keeps the dynamics
contract explicit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, overload, runtime_checkable

from stark.core.contracts.problem.dynamics import DynamicsLike
from stark.problem.dynamics.implementation import (
    DynamicsAdapterAcceptsInstantWrites,
    DynamicsAdapterAcceptsIntervalWrites,
    DynamicsAdapterAcceptsInstantReturns,
    DynamicsAdapterAcceptsIntervalReturns,
    DynamicsImplementation,
    DynamicsKernelAcceptsInstantWrites,
    DynamicsKernelAcceptsIntervalWrites,
    DynamicsKernelAcceptsInstantReturns,
    DynamicsKernelAcceptsIntervalReturns,
)
from stark.problem.dynamics.split import DynamicsSplit


@runtime_checkable
class DynamicsSignature(Protocol):
    """Recognised user dynamics declaration that can produce scheme code."""

    def implementation(self) -> DynamicsImplementation: ...


class DynamicsStyle:
    """Construct recognised dynamics signatures from user callables.

    These names are intentionally user-facing. `accepts_instant_*` variants
    receive the current time value, `accepts_interval_*` variants receive the
    whole interval object, and `*_returns` names distinguish functions that
    return translation values from functions that write into an output object.
    """

    @staticmethod
    def accepts_instant_returns(function: Callable[..., Any]) -> DynamicsSignature:
        """Recognise `function(t, state) -> translation` as a pure dynamics."""

        return DynamicsSignatureAcceptsInstantReturns(function)

    @staticmethod
    def accepts_interval_returns(function: Callable[..., Any]) -> DynamicsSignature:
        """Recognise `function(interval, state) -> translation` as a pure dynamics."""

        return DynamicsSignatureAcceptsIntervalReturns(function)

    @staticmethod
    def accepts_instant_writes(function: Callable[..., Any]) -> DynamicsSignature:
        """Recognise `function(t, state, out)` as an in-place dynamics."""

        return DynamicsSignatureAcceptsInstantWrites(function)

    @staticmethod
    def accepts_interval_writes(function: Callable[..., Any]) -> DynamicsSignature:
        """Recognise `function(interval, state, out)` as an in-place dynamics."""

        return DynamicsSignatureAcceptsIntervalWrites(function)

    @staticmethod
    def split(*, implicit: DynamicsLike, explicit: DynamicsLike) -> DynamicsSplit:
        """Recognise an implicit-explicit dynamics split."""

        return DynamicsSplit(
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
    ) -> Callable[[Callable[..., Any]], DynamicsSignatureKernelAcceptsInstantWrites]:
        ...

    @staticmethod
    @overload
    def kernel_accepts_instant_writes(
        function: Callable[..., Any],
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> DynamicsSignatureKernelAcceptsInstantWrites:
        ...

    @staticmethod
    def kernel_accepts_instant_writes(
        function: Callable[..., Any] | None = None,
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> (
        DynamicsSignatureKernelAcceptsInstantWrites
        | Callable[[Callable[..., Any]], DynamicsSignatureKernelAcceptsInstantWrites]
    ):
        """Recognise `kernel(t, *state_fields, *out_fields, *parameters)`.

        Kernel styles are for field-level array code where unpacking a full
        STARK state object would add noise or block backend acceleration. This
        writing variant receives mutable translation fields and fills them in
        place, which is the natural shape for NumPy, CuPy, and Numba kernels.
        """

        def recognise(target: Callable[..., Any]) -> DynamicsSignatureKernelAcceptsInstantWrites:
            return DynamicsSignatureKernelAcceptsInstantWrites(
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
    ) -> Callable[[Callable[..., Any]], DynamicsSignatureKernelAcceptsIntervalWrites]:
        ...

    @staticmethod
    @overload
    def kernel_accepts_interval_writes(
        function: Callable[..., Any],
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> DynamicsSignatureKernelAcceptsIntervalWrites:
        ...

    @staticmethod
    def kernel_accepts_interval_writes(
        function: Callable[..., Any] | None = None,
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> (
        DynamicsSignatureKernelAcceptsIntervalWrites
        | Callable[[Callable[..., Any]], DynamicsSignatureKernelAcceptsIntervalWrites]
    ):
        """Recognise `kernel(interval, *state_fields, *out_fields, *parameters)`.

        This is the interval-aware writing kernel form. Use it when the
        dynamics needs more than the current time value, such as the active
        step size or interval endpoint metadata.
        """

        def recognise(target: Callable[..., Any]) -> DynamicsSignatureKernelAcceptsIntervalWrites:
            return DynamicsSignatureKernelAcceptsIntervalWrites(
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
    ) -> Callable[[Callable[..., Any]], DynamicsSignatureKernelAcceptsInstantReturns]:
        ...

    @staticmethod
    @overload
    def kernel_accepts_instant_returns(
        function: Callable[..., Any],
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> DynamicsSignatureKernelAcceptsInstantReturns:
        ...

    @staticmethod
    def kernel_accepts_instant_returns(
        function: Callable[..., Any] | None = None,
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> (
        DynamicsSignatureKernelAcceptsInstantReturns
        | Callable[[Callable[..., Any]], DynamicsSignatureKernelAcceptsInstantReturns]
    ):
        """Recognise `kernel(t, *state_fields, *parameters) -> fields`.

        Returning kernels are useful for backends with immutable array values,
        notably JAX. The kernel returns field values and STARK copies them into
        the scheme translation object.
        """

        def recognise(target: Callable[..., Any]) -> DynamicsSignatureKernelAcceptsInstantReturns:
            return DynamicsSignatureKernelAcceptsInstantReturns(
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
    ) -> Callable[[Callable[..., Any]], DynamicsSignatureKernelAcceptsIntervalReturns]:
        ...

    @staticmethod
    @overload
    def kernel_accepts_interval_returns(
        function: Callable[..., Any],
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> DynamicsSignatureKernelAcceptsIntervalReturns:
        ...

    @staticmethod
    def kernel_accepts_interval_returns(
        function: Callable[..., Any] | None = None,
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> (
        DynamicsSignatureKernelAcceptsIntervalReturns
        | Callable[[Callable[..., Any]], DynamicsSignatureKernelAcceptsIntervalReturns]
    ):
        """Recognise `kernel(interval, *state_fields, *parameters) -> fields`.

        This is the interval-aware returning kernel form for immutable-array
        backends or pure expression code that also needs interval metadata.
        """

        def recognise(target: Callable[..., Any]) -> DynamicsSignatureKernelAcceptsIntervalReturns:
            return DynamicsSignatureKernelAcceptsIntervalReturns(
                function=target,
                state=state,
                translation=translation,
                parameters=tuple(parameters),
            )

        if function is None:
            return recognise
        return recognise(function)


@dataclass(frozen=True, slots=True)
class DynamicsSignatureAcceptsInstantReturns:
    """User dynamics with call shape `function(t, state) -> translation`."""

    function: Callable[..., Any]

    def implementation(self) -> DynamicsImplementation:
        return DynamicsAdapterAcceptsInstantReturns(self.function)


@dataclass(frozen=True, slots=True)
class DynamicsSignatureAcceptsIntervalReturns:
    """User dynamics with call shape `function(interval, state) -> translation`."""

    function: Callable[..., Any]

    def implementation(self) -> DynamicsImplementation:
        return DynamicsAdapterAcceptsIntervalReturns(self.function)


@dataclass(frozen=True, slots=True)
class DynamicsSignatureAcceptsInstantWrites:
    """User dynamics with call shape `function(t, state, out)`."""

    function: Callable[..., Any]

    def implementation(self) -> DynamicsImplementation:
        return DynamicsAdapterAcceptsInstantWrites(self.function)


@dataclass(frozen=True, slots=True)
class DynamicsSignatureAcceptsIntervalWrites:
    """User dynamics with call shape `function(interval, state, out)`."""

    function: Callable[..., Any]

    def implementation(self) -> DynamicsImplementation:
        return DynamicsAdapterAcceptsIntervalWrites(self.function)


@dataclass(frozen=True, slots=True)
class DynamicsSignatureKernelAcceptsInstantWrites:
    """Field-level kernel with call shape `kernel(t, *state, *out, *parameters)`."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    translation: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def with_parameters(self, *parameters: Any) -> DynamicsSignatureKernelAcceptsInstantWrites:
        """Return this recognised kernel with problem-specific parameters."""

        return DynamicsSignatureKernelAcceptsInstantWrites(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=tuple(parameters),
        )

    def implementation(self) -> DynamicsImplementation:
        return DynamicsKernelAcceptsInstantWrites(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class DynamicsSignatureKernelAcceptsIntervalWrites:
    """Field-level kernel with call shape `kernel(interval, *state, *out, *parameters)`."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    translation: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def with_parameters(self, *parameters: Any) -> DynamicsSignatureKernelAcceptsIntervalWrites:
        """Return this recognised kernel with problem-specific parameters."""

        return DynamicsSignatureKernelAcceptsIntervalWrites(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=tuple(parameters),
        )

    def implementation(self) -> DynamicsImplementation:
        return DynamicsKernelAcceptsIntervalWrites(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class DynamicsSignatureKernelAcceptsInstantReturns:
    """Field-level kernel with call shape `kernel(t, *state, *parameters) -> fields`."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    translation: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def with_parameters(self, *parameters: Any) -> DynamicsSignatureKernelAcceptsInstantReturns:
        """Return this recognised kernel with problem-specific parameters."""

        return DynamicsSignatureKernelAcceptsInstantReturns(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=tuple(parameters),
        )

    def implementation(self) -> DynamicsImplementation:
        return DynamicsKernelAcceptsInstantReturns(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class DynamicsSignatureKernelAcceptsIntervalReturns:
    """Field-level kernel with call shape `kernel(interval, *state, *parameters) -> fields`."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    translation: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def with_parameters(self, *parameters: Any) -> DynamicsSignatureKernelAcceptsIntervalReturns:
        """Return this recognised kernel with problem-specific parameters."""

        return DynamicsSignatureKernelAcceptsIntervalReturns(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=tuple(parameters),
        )

    def implementation(self) -> DynamicsImplementation:
        return DynamicsKernelAcceptsIntervalReturns(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


__all__ = [
    "DynamicsSignature",
    "DynamicsSignatureAcceptsInstantWrites",
    "DynamicsSignatureAcceptsIntervalWrites",
    "DynamicsSignatureAcceptsInstantReturns",
    "DynamicsSignatureAcceptsIntervalReturns",
    "DynamicsSignatureKernelAcceptsInstantWrites",
    "DynamicsSignatureKernelAcceptsIntervalWrites",
    "DynamicsSignatureKernelAcceptsInstantReturns",
    "DynamicsSignatureKernelAcceptsIntervalReturns",
    "DynamicsStyle",
]
