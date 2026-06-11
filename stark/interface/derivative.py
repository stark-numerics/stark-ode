from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Callable
from dataclasses import dataclass, field, fields, is_dataclass
from inspect import Parameter, signature
from typing import Any, Protocol, overload, runtime_checkable

from stark.algebraist.layout.path import AlgebraistLayoutPath
from stark.contracts.accelerator import Accelerator
from stark.contracts.derivative_imex import DerivativeIMEX


class DerivativeImplementation(Protocol):
    """Scheme-facing derivative callable prepared from a user signature."""

    def __call__(self, interval: Any, state: Any, out: Any) -> None: ...

    def accelerate(self, accelerator: Accelerator) -> "DerivativeImplementation": ...


@runtime_checkable
class DerivativeSignature(Protocol):
    """Recognised user derivative signature that can produce a scheme callable."""

    def implementation(self) -> DerivativeImplementation: ...


class DerivativeStyle:
    """Construct recognised derivative signatures from user callables."""

    @staticmethod
    def returning(function: Callable[..., Any]) -> DerivativeSignature:
        """Recognise `function(t, state) -> translation` as a pure derivative."""

        return DerivativeSignatureReturnsInstant(function)

    @staticmethod
    def interval_returning(function: Callable[..., Any]) -> DerivativeSignature:
        """Recognise `function(interval, state) -> translation` as a pure derivative."""

        return DerivativeSignatureReturnsInterval(function)

    @staticmethod
    def in_place(function: Callable[..., Any]) -> DerivativeSignature:
        """Recognise `function(t, state, out)` as an in-place derivative."""

        return DerivativeSignatureAcceptsInstant(function)

    @staticmethod
    def interval_in_place(function: Callable[..., Any]) -> DerivativeSignature:
        """Recognise `function(interval, state, out)` as an in-place derivative."""

        return DerivativeSignatureAcceptsInterval(function)

    @staticmethod
    def imex(*, implicit: object, explicit: object) -> DerivativeIMEX:
        """Recognise an implicit-explicit derivative split."""

        return DerivativeIMEX(
            implicit=implicit,
            explicit=explicit,
        )

    @staticmethod
    @overload
    def kernel(
        function: None = None,
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> Callable[[Callable[..., Any]], "DerivativeSignatureKernel"]:
        ...

    @staticmethod
    @overload
    def kernel(
        function: Callable[..., Any],
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> "DerivativeSignatureKernel":
        ...

    @staticmethod
    def kernel(
        function: Callable[..., Any] | None = None,
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> (
        "DerivativeSignatureKernel"
        " | Callable[[Callable[..., Any]], DerivativeSignatureKernel]"
    ):
        """Recognise a field-level kernel suitable for backend acceleration."""

        def recognise(target: Callable[..., Any]) -> DerivativeSignatureKernel:
            return DerivativeSignatureKernel(
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
    def kernel_returning(
        function: None = None,
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> Callable[[Callable[..., Any]], "DerivativeSignatureKernelReturning"]:
        ...

    @staticmethod
    @overload
    def kernel_returning(
        function: Callable[..., Any],
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> "DerivativeSignatureKernelReturning":
        ...

    @staticmethod
    def kernel_returning(
        function: Callable[..., Any] | None = None,
        *,
        state: tuple[str, ...],
        translation: tuple[str, ...],
        parameters: tuple[Any, ...] = (),
    ) -> (
        "DerivativeSignatureKernelReturning"
        " | Callable[[Callable[..., Any]], DerivativeSignatureKernelReturning]"
    ):
        """Recognise a pure field-level kernel returning translation values."""

        def recognise(target: Callable[..., Any]) -> DerivativeSignatureKernelReturning:
            return DerivativeSignatureKernelReturning(
                function=target,
                state=state,
                translation=translation,
                parameters=tuple(parameters),
            )

        if function is None:
            return recognise
        return recognise(function)


@dataclass(frozen=True, slots=True)
class DerivativeSignatureReturnsInstant:
    """User derivative with call shape `function(t, state) -> translation`."""

    function: Callable[..., Any]

    def implementation(self) -> DerivativeImplementation:
        return DerivativeAdapterReturnsInstant(self.function)


@dataclass(frozen=True, slots=True)
class DerivativeSignatureReturnsInterval:
    """User derivative with call shape `function(interval, state) -> translation`."""

    function: Callable[..., Any]

    def implementation(self) -> DerivativeImplementation:
        return DerivativeAdapterReturnsInterval(self.function)


@dataclass(frozen=True, slots=True)
class DerivativeSignatureAcceptsInstant:
    """User derivative with call shape `function(t, state, out)`."""

    function: Callable[..., Any]

    def implementation(self) -> DerivativeImplementation:
        return DerivativeAdapterAcceptsInstant(self.function)


@dataclass(frozen=True, slots=True)
class DerivativeSignatureAcceptsInterval:
    """User derivative with call shape `function(interval, state, out)`."""

    function: Callable[..., Any]

    def implementation(self) -> DerivativeImplementation:
        return DerivativeAdapterAcceptsInterval(self.function)


@dataclass(frozen=True, slots=True)
class DerivativeSignatureKernel:
    """Field-level derivative kernel with explicit state and translation fields."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    translation: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def with_parameters(self, *parameters: Any) -> "DerivativeSignatureKernel":
        """Return this recognised kernel with problem-specific parameters."""

        return DerivativeSignatureKernel(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=tuple(parameters),
        )

    def implementation(self) -> DerivativeImplementation:
        return DerivativeKernel(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class DerivativeSignatureKernelReturning:
    """Field-level derivative kernel returning translation field values."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    translation: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def with_parameters(self, *parameters: Any) -> "DerivativeSignatureKernelReturning":
        """Return this recognised kernel with problem-specific parameters."""

        return DerivativeSignatureKernelReturning(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=tuple(parameters),
        )

    def implementation(self) -> DerivativeImplementation:
        return DerivativeKernelReturning(
            function=self.function,
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class Derivative:
    """Canonical derivative declaration accepted by `System`."""

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
            return DerivativeStyle.returning(function)
        if len(positional) == 3:
            return DerivativeStyle.in_place(function)

        raise TypeError(
            "Plain derivative callables must use function(t, state) or "
            "function(t, state, out). Use DerivativeStyle.interval_returning(...) "
            "or DerivativeStyle.interval_in_place(...) for interval-aware derivatives."
        )

    @classmethod
    def returning(cls, function: Callable[..., Any]) -> DerivativeSignature:
        """Declare a pure derivative `function(t, state) -> translation`."""

        return DerivativeStyle.returning(function)

    @classmethod
    def interval_returning(cls, function: Callable[..., Any]) -> DerivativeSignature:
        """Declare a pure derivative `function(interval, state) -> translation`."""

        return DerivativeStyle.interval_returning(function)

    @classmethod
    def in_place(cls, function: Callable[..., Any]) -> DerivativeSignature:
        """Declare an in-place derivative `function(t, state, out)`."""

        return DerivativeStyle.in_place(function)

    @classmethod
    def interval_in_place(cls, function: Callable[..., Any]) -> DerivativeSignature:
        """Declare an in-place derivative `function(interval, state, out)`."""

        return DerivativeStyle.interval_in_place(function)

    @classmethod
    def imex(cls, *, implicit: object, explicit: object) -> DerivativeIMEX:
        """Declare an implicit-explicit derivative split."""

        return DerivativeStyle.imex(implicit=implicit, explicit=explicit)

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.implementation(interval, state, out)

    def accelerate(self, accelerator: Accelerator) -> DerivativeImplementation:
        return self.implementation.accelerate(accelerator)


@dataclass(frozen=True, slots=True)
class DerivativeAdapterAcceptsInstant:
    """Scheme derivative adapter for `function(t, state, out)`."""

    function: Callable[..., Any]

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.function(interval.present, state, out)

    def accelerate(self, accelerator: Accelerator) -> "DerivativeAdapterAcceptsInstant":
        del accelerator
        return self


@dataclass(frozen=True, slots=True)
class DerivativeAdapterAcceptsInterval:
    """Scheme derivative adapter for `function(interval, state, out)`."""

    function: Callable[..., Any]

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.function(interval, state, out)

    def accelerate(self, accelerator: Accelerator) -> "DerivativeAdapterAcceptsInterval":
        del accelerator
        return self


@dataclass(frozen=True, slots=True)
class DerivativeAdapterReturnsInstant:
    """Scheme derivative adapter for `function(t, state) -> translation`."""

    function: Callable[..., Any]

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        assign_returned_translation(self.function(interval.present, state), out)

    def accelerate(self, accelerator: Accelerator) -> "DerivativeAdapterReturnsInstant":
        del accelerator
        return self


@dataclass(frozen=True, slots=True)
class DerivativeAdapterReturnsInterval:
    """Scheme derivative adapter for `function(interval, state) -> translation`."""

    function: Callable[..., Any]

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        assign_returned_translation(self.function(interval, state), out)

    def accelerate(self, accelerator: Accelerator) -> "DerivativeAdapterReturnsInterval":
        del accelerator
        return self


@dataclass(frozen=True, slots=True)
class DerivativeKernel:
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

    def accelerate(self, accelerator: Accelerator) -> "DerivativeKernel":
        return DerivativeKernel(
            function=accelerator.compile(self.function),
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class DerivativeKernelReturning:
    """Scheme derivative adapter for a pure field-level kernel."""

    function: Callable[..., Any]
    state: tuple[str, ...]
    translation: tuple[str, ...]
    parameters: tuple[Any, ...] = ()

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        del interval
        result = self.function(
            *(getattr(state, name) for name in self.state),
            *self.parameters,
        )
        assign_returned_fields(result, out, self.translation)

    def accelerate(self, accelerator: Accelerator) -> "DerivativeKernelReturning":
        return DerivativeKernelReturning(
            function=accelerator.compile(self.function),
            state=self.state,
            translation=self.translation,
            parameters=self.parameters,
        )


def assign_returned_translation(result: Any, out: Any) -> None:
    """Copy a return-style derivative result into the scheme translation object."""

    if result is None:
        raise TypeError("Return-style derivatives must return translation values.")

    layout = getattr(out, "algebraist_layout", None)
    fields_ = getattr(layout, "fields", None)
    if fields_ is not None:
        field_tuple = tuple(fields_)
        values = returned_values_for_layout(result, field_tuple)
        for field_, value in zip(field_tuple, values, strict=True):
            field_.translation_path.set(out, value)
        return

    assign_returned_without_layout(result, out)


def assign_returned_fields(result: Any, out: Any, paths: tuple[str, ...]) -> None:
    """Copy a pure field-kernel result into explicit translation paths."""

    if result is None:
        raise TypeError("Return-style kernels must return translation values.")

    normalized = tuple(AlgebraistLayoutPath(path) for path in paths)
    if isinstance(result, Mapping):
        values = [mapping_value(result, path, path) for path in normalized]
    elif len(normalized) == 1:
        values = [result]
    elif isinstance(result, tuple | list) and len(result) == len(normalized):
        values = list(result)
    else:
        values = [object_value(result, path, path) for path in normalized]

    for path, value in zip(normalized, values, strict=True):
        path.set(out, value)


def returned_values_for_layout(result: Any, fields_: tuple[Any, ...]) -> tuple[Any, ...]:
    """Return translation values from a user result, aligned to layout fields."""

    if isinstance(result, Mapping):
        return tuple(
            mapping_value(result, field_.translation_path, field_.state_path)
            for field_ in fields_
        )
    if len(fields_) == 1:
        field_ = fields_[0]
        value = object_value_or_missing(result, field_.translation_path, field_.state_path)
        if value is not _MISSING:
            return (value,)
        return (result,)
    if isinstance(result, tuple | list) and len(result) == len(fields_):
        return tuple(result)
    return tuple(
        object_value(result, field_.translation_path, field_.state_path)
        for field_ in fields_
    )


def mapping_value(
    result: Mapping[Any, Any],
    translation_path: AlgebraistLayoutPath,
    state_path: AlgebraistLayoutPath,
) -> Any:
    """Read one layout field from a mapping-style derivative return value."""

    for key in path_keys(translation_path, state_path):
        if key in result:
            return result[key]
    raise KeyError(
        "Return-style derivative did not provide translation field "
        f"{translation_path!s}."
    )


def object_value(
    result: Any,
    translation_path: AlgebraistLayoutPath,
    state_path: AlgebraistLayoutPath,
) -> Any:
    """Read one layout field from an object-style derivative return value."""

    value = object_value_or_missing(result, translation_path, state_path)
    if value is _MISSING:
        raise AttributeError(
            "Return-style derivative did not provide translation field "
            f"{translation_path!s}."
        )
    return value


def object_value_or_missing(
    result: Any,
    translation_path: AlgebraistLayoutPath,
    state_path: AlgebraistLayoutPath,
) -> Any:
    """Read one layout field from an object-style result if present."""

    for path in (translation_path, state_path):
        try:
            return path.get(result)
        except AttributeError:
            continue
    return _MISSING


def path_keys(
    translation_path: AlgebraistLayoutPath,
    state_path: AlgebraistLayoutPath,
) -> tuple[Any, ...]:
    """Mapping keys accepted for one layout field."""

    keys: list[Any] = []
    for path in (translation_path, state_path):
        keys.extend((str(path), path.name, path.parts))
    return tuple(dict.fromkeys(keys))


def assign_returned_without_layout(result: Any, out: Any) -> None:
    """Best-effort return assignment for tests and simple ad-hoc translations."""

    if isinstance(result, Mapping):
        for key, value in result.items():
            if isinstance(key, str) and key.isidentifier():
                setattr(out, key, value)
            else:
                raise TypeError(
                    "Mapping-style derivative returns require string identifier keys."
                )
        return

    names = writable_field_names(out)
    if len(names) == 1:
        setattr(out, names[0], result)
        return

    raise TypeError(
        "Return-style derivative assignment requires a layout-backed translation "
        "object, a mapping return value, or an output object with one writable field."
    )


def writable_field_names(out: Any) -> tuple[str, ...]:
    """Return likely writable public field names for a translation-like object."""

    if is_dataclass(out):
        return tuple(field_.name for field_ in fields(out) if field_.init)

    slots = getattr(type(out), "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    if slots:
        return tuple(name for name in slots if not name.startswith("_"))

    dictionary = getattr(out, "__dict__", None)
    if dictionary is not None:
        return tuple(name for name in dictionary if not name.startswith("_"))

    return ()


_MISSING = object()


__all__ = [
    "Derivative",
    "DerivativeImplementation",
    "DerivativeAdapterAcceptsInterval",
    "DerivativeAdapterReturnsInterval",
    "DerivativeAdapterReturnsInstant",
    "DerivativeKernel",
    "DerivativeKernelReturning",
    "DerivativeSignature",
    "DerivativeSignatureAcceptsInterval",
    "DerivativeSignatureKernel",
    "DerivativeSignatureKernelReturning",
    "DerivativeSignatureAcceptsInstant",
    "DerivativeSignatureReturnsInterval",
    "DerivativeSignatureReturnsInstant",
    "DerivativeStyle",
    "DerivativeAdapterAcceptsInstant",
    "assign_returned_translation",
]
