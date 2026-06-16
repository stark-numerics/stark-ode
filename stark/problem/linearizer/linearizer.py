from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import Any, Protocol, overload, runtime_checkable

from stark.core.contracts.accelerator import Accelerator
from stark.problem.derivative.derivative import assign_returned_fields


class LinearizerImplementation(Protocol):
    """Resolvent-facing linearizer callable prepared from a user signature."""

    def __call__(self, interval: Any, state: Any, out: Any) -> None: ...

    def accelerate(self, accelerator: Accelerator) -> "LinearizerImplementation": ...


@runtime_checkable
class LinearizerSignature(Protocol):
    """Recognised user linearizer signature that can produce a resolvent callable."""

    def implementation(self) -> LinearizerImplementation: ...


class LinearizerStyle:
    """Construct recognised linearizer signatures from user callables.

    A linearizer is the problem-level worker that configures the local Jacobian
    operator used by implicit resolvents. Its prepared form always satisfies the
    lower-level ``LinearizerLike`` contract:

        linearizer(interval, state, out) -> None

    where ``out`` is a mutable operator container. Styles only describe how a
    user wants to provide that operation.
    """

    @staticmethod
    def in_place(function: Callable[..., Any]) -> LinearizerSignature:
        """Recognise ``function(t, state, out)`` as an in-place linearizer."""

        return LinearizerSignatureAcceptsInstant(function)

    @staticmethod
    def interval_in_place(function: Callable[..., Any]) -> LinearizerSignature:
        """Recognise ``function(interval, state, out)`` as an in-place linearizer."""

        return LinearizerSignatureAcceptsInterval(function)

    @staticmethod
    @overload
    def kernel(
        function: None = None,
        *,
        state: tuple[str, ...],
        source: tuple[str, ...],
        target: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> Callable[[Callable[..., Any]], "LinearizerSignatureKernel"]:
        ...

    @staticmethod
    @overload
    def kernel(
        function: Callable[..., Any],
        *,
        state: tuple[str, ...],
        source: tuple[str, ...],
        target: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> "LinearizerSignatureKernel":
        ...

    @staticmethod
    def kernel(
        function: Callable[..., Any] | None = None,
        *,
        state: tuple[str, ...],
        source: tuple[str, ...],
        target: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> "LinearizerSignatureKernel | Callable[[Callable[..., Any]], LinearizerSignatureKernel]":
        """Recognise a field-level Jacobian-action kernel.

        The wrapped kernel receives state fields, source translation fields,
        target translation fields, then any parameters::

            kernel(*(state fields), *(source fields), *(target fields), *parameters)

        The prepared linearizer installs this kernel as ``out.apply``.
        """

        def recognise(target_function: Callable[..., Any]) -> LinearizerSignatureKernel:
            return LinearizerSignatureKernel(
                function=target_function,
                state=tuple(state),
                source=tuple(source),
                target=tuple(target),
                parameters=tuple(parameters),
            )

        if function is None:
            return recognise
        return recognise(function)

    @staticmethod
    @overload
    def kernel_returning(
        function: None = None,
        *,
        state: tuple[str, ...],
        source: tuple[str, ...],
        target: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> Callable[[Callable[..., Any]], "LinearizerSignatureKernelReturning"]:
        ...

    @staticmethod
    @overload
    def kernel_returning(
        function: Callable[..., Any],
        *,
        state: tuple[str, ...],
        source: tuple[str, ...],
        target: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> "LinearizerSignatureKernelReturning":
        ...

    @staticmethod
    def kernel_returning(
        function: Callable[..., Any] | None = None,
        *,
        state: tuple[str, ...],
        source: tuple[str, ...],
        target: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> (
        "LinearizerSignatureKernelReturning"
        " | Callable[[Callable[..., Any]], LinearizerSignatureKernelReturning]"
    ):
        """Recognise a pure field-level Jacobian-action kernel."""

        def recognise(target_function: Callable[..., Any]) -> LinearizerSignatureKernelReturning:
            return LinearizerSignatureKernelReturning(
                function=target_function,
                state=tuple(state),
                source=tuple(source),
                target=tuple(target),
                parameters=tuple(parameters),
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
    ) -> Callable[[Callable[..., Any]], "LinearizerSignatureDense"]:
        ...

    @staticmethod
    @overload
    def dense(
        function: Callable[..., Any],
        *,
        state: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> "LinearizerSignatureDense":
        ...

    @staticmethod
    def dense(
        function: Callable[..., Any] | None = None,
        *,
        state: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> "LinearizerSignatureDense | Callable[[Callable[..., Any]], LinearizerSignatureDense]":
        """Recognise a field-level dense Jacobian fill kernel.

        The kernel receives state fields, then ``matrix, row_offset,
        column_offset, stride``, then any parameters. It fills a flat row-major
        matrix buffer in place. The translation basis is accepted by the
        operator contract but intentionally not passed to this field-level
        kernel; basis-specific dense fills should use ``interval_in_place``.
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
        state: tuple[str, ...],
        source: tuple[str, ...],
        target: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
        dense_parameters: tuple[Any, ...] | None = None,
    ) -> LinearizerSignature:
        """Combine field-level apply and dense-fill kernels into one linearizer.

        ``apply`` uses the same convention as ``LinearizerStyle.kernel``.
        ``dense`` uses the same convention as ``LinearizerStyle.dense``. At
        least one must be supplied. Dense inverters use ``dense_fill`` when it
        is available; Krylov-style inverters can use only ``apply``.
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
    """Field-level Jacobian-action kernel signature."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    source: tuple[str, ...]
    target: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def with_parameters(self, *parameters: Any) -> "LinearizerSignatureKernel":
        return LinearizerSignatureKernel(
            function=self.function,
            state=self.state,
            source=self.source,
            target=self.target,
            parameters=tuple(parameters),
        )

    def implementation(self) -> LinearizerImplementation:
        return LinearizerKernel(
            function=self.function,
            state=self.state,
            source=self.source,
            target=self.target,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class LinearizerSignatureKernelReturning:
    """Pure field-level Jacobian-action kernel signature."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    source: tuple[str, ...]
    target: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def with_parameters(self, *parameters: Any) -> "LinearizerSignatureKernelReturning":
        return LinearizerSignatureKernelReturning(
            function=self.function,
            state=self.state,
            source=self.source,
            target=self.target,
            parameters=tuple(parameters),
        )

    def implementation(self) -> LinearizerImplementation:
        return LinearizerKernelReturning(
            function=self.function,
            state=self.state,
            source=self.source,
            target=self.target,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class LinearizerSignatureDense:
    """Field-level dense Jacobian fill kernel signature."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def with_parameters(self, *parameters: Any) -> "LinearizerSignatureDense":
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


@dataclass(frozen=True, slots=True)
class Linearizer:
    """Canonical linearizer declaration accepted by ``System``."""

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
        ``LinearizerStyle.in_place`` explicitly for ``function(t, state, out)``.
        """

        try:
            inspected = signature(function)
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
            return LinearizerStyle.interval_in_place(function)

        raise TypeError(
            "Plain linearizer callables must use function(interval, state, out). "
            "Use LinearizerStyle for field-level kernels or instant-time callables."
        )

    @classmethod
    def in_place(cls, function: Callable[..., Any]) -> LinearizerSignature:
        return LinearizerStyle.in_place(function)

    @classmethod
    def interval_in_place(cls, function: Callable[..., Any]) -> LinearizerSignature:
        return LinearizerStyle.interval_in_place(function)

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.implementation(interval, state, out)

    def accelerate(self, accelerator: Accelerator) -> LinearizerImplementation:
        return self.implementation.accelerate(accelerator)


@dataclass(frozen=True, slots=True)
class LinearizerAdapterAcceptsInstant:
    """Resolvent linearizer adapter for ``function(t, state, out)``."""

    function: Callable[..., Any]

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.function(interval.present, state, out)

    def accelerate(self, accelerator: Accelerator) -> "LinearizerAdapterAcceptsInstant":
        del accelerator
        return self


@dataclass(frozen=True, slots=True)
class LinearizerAdapterAcceptsInterval:
    """Resolvent linearizer adapter for ``function(interval, state, out)``."""

    function: Callable[..., Any]

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.function(interval, state, out)

    def accelerate(self, accelerator: Accelerator) -> "LinearizerAdapterAcceptsInterval":
        del accelerator
        return self


@dataclass(frozen=True, slots=True)
class LinearizerKernel:
    """Prepared field-level Jacobian-action kernel."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    source: tuple[str, ...]
    target: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        del interval
        state_values = tuple(getattr(state, name) for name in self.state)
        function = self.function
        source_names = self.source
        target_names = self.target
        parameters = self.parameters

        def apply(source: Any, result: Any) -> None:
            function(
                *state_values,
                *(getattr(source, name) for name in source_names),
                *(getattr(result, name) for name in target_names),
                *parameters,
            )

        out.apply = apply

    def accelerate(self, accelerator: Accelerator) -> "LinearizerKernel":
        return LinearizerKernel(
            function=accelerator.compile(self.function, label="linearizer-apply", cache=True),
            state=self.state,
            source=self.source,
            target=self.target,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class LinearizerKernelReturning:
    """Prepared pure field-level Jacobian-action kernel."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    source: tuple[str, ...]
    target: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        del interval
        state_values = tuple(getattr(state, name) for name in self.state)
        function = self.function
        source_names = self.source
        target_names = self.target
        parameters = self.parameters

        def apply(source: Any, result: Any) -> None:
            values = function(
                *state_values,
                *(getattr(source, name) for name in source_names),
                *parameters,
            )
            assign_returned_fields(values, result, target_names)

        out.apply = apply

    def accelerate(self, accelerator: Accelerator) -> "LinearizerKernelReturning":
        return LinearizerKernelReturning(
            function=accelerator.compile(self.function, label="linearizer-apply-returning", cache=True),
            state=self.state,
            source=self.source,
            target=self.target,
            parameters=self.parameters,
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

    def accelerate(self, accelerator: Accelerator) -> "LinearizerDense":
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

    def accelerate(self, accelerator: Accelerator) -> "LinearizerOperator":
        return LinearizerOperator(
            apply=self.apply.accelerate(accelerator) if self.apply is not None else None,
            dense=self.dense.accelerate(accelerator) if self.dense is not None else None,
        )



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
        return LinearizerSignatureDense(
            function=candidate,
            state=tuple(state),
            parameters=tuple(parameters),
        )
    raise TypeError("LinearizerStyle.operator dense must be callable or a linearizer signature.")


__all__ = [
    "Linearizer",
    "LinearizerImplementation",
    "LinearizerSignature",
    "LinearizerStyle",
]
