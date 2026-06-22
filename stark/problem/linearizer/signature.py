"""User-facing linearizer declarations.

Signature objects record the shape of a user linearizer and any metadata needed
to turn it into resolvent-facing code. Linearizers configure local Jacobian
operators for implicit methods; their prepared form always satisfies
``linearizer(interval, state, out) -> None``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, overload, runtime_checkable

from stark.problem.linearizer.implementation import (
    LinearizerAdapterAcceptsInstant,
    LinearizerAdapterAcceptsInterval,
    LinearizerDense,
    LinearizerImplementation,
    LinearizerKernel,
    LinearizerKernelReturning,
    LinearizerOperator,
)


@runtime_checkable
class LinearizerSignature(Protocol):
    """Recognised user linearizer declaration that can produce resolvent code."""

    def implementation(self) -> LinearizerImplementation: ...


class LinearizerStyle:
    """Construct recognised linearizer signatures from user callables.

    A plain linearizer configures an operator object directly. Kernel styles are
    for field-level Jacobian-action code: writing kernels fill target
    translation fields in place, while returning kernels compute target field
    values and return them. Instant variants receive the current time value;
    interval variants receive the full interval object.
    """

    @staticmethod
    def accepts_instant_writes(function: Callable[..., Any]) -> LinearizerSignature:
        """Recognise ``function(t, state, out)`` as an in-place linearizer."""

        return LinearizerSignatureAcceptsInstant(function)

    @staticmethod
    def accepts_interval_writes(function: Callable[..., Any]) -> LinearizerSignature:
        """Recognise ``function(interval, state, out)`` as an in-place linearizer."""

        return LinearizerSignatureAcceptsInterval(function)

    @staticmethod
    @overload
    def kernel_accepts_instant_writes(
        function: None = None,
        *,
        state: tuple[str, ...],
        source: tuple[str, ...],
        target: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> Callable[[Callable[..., Any]], LinearizerSignatureKernel]:
        ...

    @staticmethod
    @overload
    def kernel_accepts_instant_writes(
        function: Callable[..., Any],
        *,
        state: tuple[str, ...],
        source: tuple[str, ...],
        target: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> LinearizerSignatureKernel:
        ...

    @staticmethod
    def kernel_accepts_instant_writes(
        function: Callable[..., Any] | None = None,
        *,
        state: tuple[str, ...],
        source: tuple[str, ...],
        target: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> LinearizerSignatureKernel | Callable[[Callable[..., Any]], LinearizerSignatureKernel]:
        """Recognise ``kernel(t, *state, *source, *target, *parameters)``.

        The prepared linearizer installs this kernel as ``out.apply``. Use this
        form for mutable-array Jacobian actions where backend acceleration
        benefits from explicit field arguments.
        """

        def recognise(target_function: Callable[..., Any]) -> LinearizerSignatureKernel:
            return LinearizerSignatureKernel(
                function=target_function,
                state=tuple(state),
                source=tuple(source),
                target=tuple(target),
                parameters=tuple(parameters),
                accepts_interval=False,
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
        source: tuple[str, ...],
        target: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> Callable[[Callable[..., Any]], LinearizerSignatureKernel]:
        ...

    @staticmethod
    @overload
    def kernel_accepts_interval_writes(
        function: Callable[..., Any],
        *,
        state: tuple[str, ...],
        source: tuple[str, ...],
        target: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> LinearizerSignatureKernel:
        ...

    @staticmethod
    def kernel_accepts_interval_writes(
        function: Callable[..., Any] | None = None,
        *,
        state: tuple[str, ...],
        source: tuple[str, ...],
        target: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> LinearizerSignatureKernel | Callable[[Callable[..., Any]], LinearizerSignatureKernel]:
        """Recognise ``kernel(interval, *state, *source, *target, *parameters)``."""

        def recognise(target_function: Callable[..., Any]) -> LinearizerSignatureKernel:
            return LinearizerSignatureKernel(
                function=target_function,
                state=tuple(state),
                source=tuple(source),
                target=tuple(target),
                parameters=tuple(parameters),
                accepts_interval=True,
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
        source: tuple[str, ...],
        target: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> Callable[[Callable[..., Any]], LinearizerSignatureKernelReturning]:
        ...

    @staticmethod
    @overload
    def kernel_accepts_instant_returns(
        function: Callable[..., Any],
        *,
        state: tuple[str, ...],
        source: tuple[str, ...],
        target: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> LinearizerSignatureKernelReturning:
        ...

    @staticmethod
    def kernel_accepts_instant_returns(
        function: Callable[..., Any] | None = None,
        *,
        state: tuple[str, ...],
        source: tuple[str, ...],
        target: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> LinearizerSignatureKernelReturning | Callable[[Callable[..., Any]], LinearizerSignatureKernelReturning]:
        """Recognise ``kernel(t, *state, *source, *parameters) -> target``."""

        def recognise(target_function: Callable[..., Any]) -> LinearizerSignatureKernelReturning:
            return LinearizerSignatureKernelReturning(
                function=target_function,
                state=tuple(state),
                source=tuple(source),
                target=tuple(target),
                parameters=tuple(parameters),
                accepts_interval=False,
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
        source: tuple[str, ...],
        target: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> Callable[[Callable[..., Any]], LinearizerSignatureKernelReturning]:
        ...

    @staticmethod
    @overload
    def kernel_accepts_interval_returns(
        function: Callable[..., Any],
        *,
        state: tuple[str, ...],
        source: tuple[str, ...],
        target: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> LinearizerSignatureKernelReturning:
        ...

    @staticmethod
    def kernel_accepts_interval_returns(
        function: Callable[..., Any] | None = None,
        *,
        state: tuple[str, ...],
        source: tuple[str, ...],
        target: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> LinearizerSignatureKernelReturning | Callable[[Callable[..., Any]], LinearizerSignatureKernelReturning]:
        """Recognise ``kernel(interval, *state, *source, *parameters) -> target``."""

        def recognise(target_function: Callable[..., Any]) -> LinearizerSignatureKernelReturning:
            return LinearizerSignatureKernelReturning(
                function=target_function,
                state=tuple(state),
                source=tuple(source),
                target=tuple(target),
                parameters=tuple(parameters),
                accepts_interval=True,
            )

        if function is None:
            return recognise
        return recognise(function)

    @staticmethod
    @overload
    def dense(
        function: None = None,
        *,
        state: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> Callable[[Callable[..., Any]], LinearizerSignatureDense]:
        ...

    @staticmethod
    @overload
    def dense(
        function: Callable[..., Any],
        *,
        state: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> LinearizerSignatureDense:
        ...

    @staticmethod
    def dense(
        function: Callable[..., Any] | None = None,
        *,
        state: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> LinearizerSignatureDense | Callable[[Callable[..., Any]], LinearizerSignatureDense]:
        """Recognise a field-level dense Jacobian fill kernel.

        The kernel receives state fields, then ``matrix, row_offset,
        column_offset, stride``, then any parameters. It fills a flat row-major
        matrix buffer in place.
        """

        def recognise(target_function: Callable[..., Any]) -> LinearizerSignatureDense:
            return LinearizerSignatureDense(
                function=target_function,
                state=tuple(state),
                parameters=tuple(parameters),
            )

        if function is None:
            return recognise
        return recognise(function)

    @staticmethod
    def operator(
        *,
        apply: Callable[..., Any] | LinearizerSignature | None = None,
        dense: Callable[..., Any] | LinearizerSignature | None = None,
        state: tuple[str, ...] = (),
        source: tuple[str, ...] = (),
        target: tuple[str, ...] = (),
        parameters: tuple[Any, ...] = (),
        dense_parameters: tuple[Any, ...] | None = None,
    ) -> LinearizerSignature:
        """Combine field-level apply and dense-fill kernels into one linearizer.

        ``apply`` uses the same convention as
        ``LinearizerStyle.kernel_accepts_instant_writes``. Dense inverters use
        ``dense_fill`` when it is available; Krylov-style inverters can use
        only ``apply``.

        When `apply` and `dense` are already `LinearizerStyle` signatures, the
        field metadata lives on those signatures and does not need to be passed
        again. Raw callables still need `state`, `source`, and `target` so STARK
        can build the corresponding signatures.
        """

        if apply is None and dense is None:
            raise TypeError("LinearizerStyle.operator requires apply, dense, or both.")

        apply_signature = _coerce_apply_signature(
            apply,
            state=state,
            source=source,
            target=target,
            parameters=parameters,
        )
        dense_signature = _coerce_dense_signature(
            dense,
            state=state,
            parameters=parameters if dense_parameters is None else dense_parameters,
        )
        return LinearizerSignatureOperator(apply=apply_signature, dense=dense_signature)


@dataclass(frozen=True, slots=True)
class LinearizerSignatureAcceptsInstant:
    """User linearizer with call shape ``function(t, state, out)``."""

    function: Callable[..., Any]

    def implementation(self) -> LinearizerImplementation:
        return LinearizerAdapterAcceptsInstant(self.function)


@dataclass(frozen=True, slots=True)
class LinearizerSignatureAcceptsInterval:
    """User linearizer with call shape ``function(interval, state, out)``."""

    function: Callable[..., Any]

    def implementation(self) -> LinearizerImplementation:
        return LinearizerAdapterAcceptsInterval(self.function)


@dataclass(frozen=True, slots=True)
class LinearizerSignatureKernel:
    """Field-level Jacobian-action kernel signature that writes target fields."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    source: tuple[str, ...]
    target: tuple[str, ...]
    parameters: tuple[Any, ...] = ()
    accepts_interval: bool = False

    def with_parameters(self, *parameters: Any) -> LinearizerSignatureKernel:
        return LinearizerSignatureKernel(
            function=self.function,
            state=self.state,
            source=self.source,
            target=self.target,
            parameters=tuple(parameters),
            accepts_interval=self.accepts_interval,
        )

    def implementation(self) -> LinearizerImplementation:
        return LinearizerKernel(
            function=self.function,
            state=self.state,
            source=self.source,
            target=self.target,
            parameters=self.parameters,
            accepts_interval=self.accepts_interval,
        )


@dataclass(frozen=True, slots=True)
class LinearizerSignatureKernelReturning:
    """Field-level Jacobian-action kernel signature that returns target fields."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    source: tuple[str, ...]
    target: tuple[str, ...]
    parameters: tuple[Any, ...] = ()
    accepts_interval: bool = False

    def with_parameters(self, *parameters: Any) -> LinearizerSignatureKernelReturning:
        return LinearizerSignatureKernelReturning(
            function=self.function,
            state=self.state,
            source=self.source,
            target=self.target,
            parameters=tuple(parameters),
            accepts_interval=self.accepts_interval,
        )

    def implementation(self) -> LinearizerImplementation:
        return LinearizerKernelReturning(
            function=self.function,
            state=self.state,
            source=self.source,
            target=self.target,
            parameters=self.parameters,
            accepts_interval=self.accepts_interval,
        )


@dataclass(frozen=True, slots=True)
class LinearizerSignatureDense:
    """Field-level dense Jacobian fill kernel signature."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def with_parameters(self, *parameters: Any) -> LinearizerSignatureDense:
        return LinearizerSignatureDense(
            function=self.function,
            state=self.state,
            parameters=tuple(parameters),
        )

    def implementation(self) -> LinearizerImplementation:
        return LinearizerDense(
            function=self.function,
            state=self.state,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class LinearizerSignatureOperator:
    """Combined apply/dense linearizer declaration."""

    apply: LinearizerSignatureKernel | LinearizerSignatureKernelReturning | None = None
    dense: LinearizerSignatureDense | None = None

    def implementation(self) -> LinearizerImplementation:
        apply_impl = self.apply.implementation() if self.apply is not None else None
        dense_impl = self.dense.implementation() if self.dense is not None else None
        return LinearizerOperator(apply=apply_impl, dense=dense_impl)


def _coerce_apply_signature(
    candidate: Callable[..., Any] | LinearizerSignature | None,
    *,
    state: tuple[str, ...],
    source: tuple[str, ...],
    target: tuple[str, ...],
    parameters: tuple[Any, ...],
) -> LinearizerSignatureKernel | LinearizerSignatureKernelReturning | None:
    if candidate is None:
        return None
    if isinstance(candidate, LinearizerSignature):
        if isinstance(candidate, (LinearizerSignatureKernel, LinearizerSignatureKernelReturning)):
            return candidate
        raise TypeError("LinearizerStyle.operator apply must be a kernel-style signature.")
    if callable(candidate):
        if not state or not source or not target:
            raise TypeError(
                "LinearizerStyle.operator raw apply callables require "
                "state, source, and target field names."
            )
        return LinearizerSignatureKernel(
            function=candidate,
            state=tuple(state),
            source=tuple(source),
            target=tuple(target),
            parameters=tuple(parameters),
        )
    raise TypeError("LinearizerStyle.operator apply must be callable or a linearizer signature.")


def _coerce_dense_signature(
    candidate: Callable[..., Any] | LinearizerSignature | None,
    *,
    state: tuple[str, ...],
    parameters: tuple[Any, ...],
) -> LinearizerSignatureDense | None:
    if candidate is None:
        return None
    if isinstance(candidate, LinearizerSignature):
        if isinstance(candidate, LinearizerSignatureDense):
            return candidate
        raise TypeError("LinearizerStyle.operator dense must be a dense linearizer signature.")
    if callable(candidate):
        if not state:
            raise TypeError(
                "LinearizerStyle.operator raw dense callables require state field names."
            )
        return LinearizerSignatureDense(
            function=candidate,
            state=tuple(state),
            parameters=tuple(parameters),
        )
    raise TypeError("LinearizerStyle.operator dense must be callable or a linearizer signature.")


__all__ = [
    "LinearizerSignature",
    "LinearizerSignatureAcceptsInstant",
    "LinearizerSignatureAcceptsInterval",
    "LinearizerSignatureDense",
    "LinearizerSignatureKernel",
    "LinearizerSignatureKernelReturning",
    "LinearizerSignatureOperator",
    "LinearizerStyle",
]
