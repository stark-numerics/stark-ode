from dataclasses import dataclass
from typing import Any, Callable, Protocol
import inspect

from stark.contracts import Carrier


class DerivativeConvention(Protocol):
    def __call__(
        self,
        function: Callable[..., Any],
        t: Any,
        y: Any,
        dy: Any,
        carrier: Carrier[Any, Any],
    ) -> Any: ...


class DerivativeConventionReturn:
    def __call__(
        self,
        function: Callable[..., Any],
        t: Any,
        y: Any,
        dy: Any,
        carrier: Carrier[Any, Any],
    ) -> Any:
        value = function(t, y)
        return carrier.validation.coerce_translation(value)


class DerivativeConventionInPlace:
    def __call__(
        self,
        function: Callable[..., Any],
        t: Any,
        y: Any,
        dy: Any,
        carrier: Carrier[Any, Any],
    ) -> Any:
        function(t, y, dy)
        return carrier.validation.validate_translation(dy)


@dataclass(slots=True)
class StarkDerivative:
    function: Callable[..., Any]
    convention: DerivativeConvention | None = None

    def bind(self, carrier: Carrier[Any, Any]) -> Any:
        convention = self.resolve_convention(carrier)

        if isinstance(convention, DerivativeConventionReturn):
            return DerivativeRuntimeReturn(self.function, carrier)

        if isinstance(convention, DerivativeConventionInPlace):
            return DerivativeRuntimeInPlace(self.function, carrier)

        return DerivativeRuntimeCustom(
            function=self.function,
            convention=convention,
            carrier=carrier,
        )

    def resolve_convention(self, carrier: Carrier[Any, Any]) -> DerivativeConvention:
        if self.convention is not None:
            return self.convention

        if carrier.arithmetic.preference == "into":
            return DerivativeConventionInPlace()

        return DerivativeConventionReturn()

    @classmethod
    def returning(cls, function: Callable[..., Any]) -> "StarkDerivative":
        return cls(function=function, convention=DerivativeConventionReturn())

    @classmethod
    def in_place(cls, function: Callable[..., Any]) -> "StarkDerivative":
        return cls(function=function, convention=DerivativeConventionInPlace())

    @classmethod
    def from_callable(cls, function: Callable[..., Any]) -> "StarkDerivative":
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            return cls(function=function, convention=None)

        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and parameter.default is inspect.Parameter.empty
        ]

        if len(positional) == 2:
            return cls.returning(function)

        if len(positional) == 3:
            return cls.in_place(function)

        return cls(function=function, convention=None)


@dataclass(slots=True)
class DerivativeRuntimeReturn:
    function: Callable[..., Any]
    carrier: Carrier[Any, Any]

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        out.value = self.carrier.validation.coerce_translation(
            self.function(interval.present, state.value)
        )


@dataclass(slots=True)
class DerivativeRuntimeInPlace:
    function: Callable[..., Any]
    carrier: Carrier[Any, Any]

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.function(interval.present, state.value, out.value)
        self.carrier.validation.validate_translation(out.value)


@dataclass(slots=True)
class DerivativeRuntimeCustom:
    function: Callable[..., Any]
    convention: DerivativeConvention
    carrier: Carrier[Any, Any]

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        out.value = self.convention(
            self.function,
            interval.present,
            state.value,
            out.value,
            self.carrier,
        )