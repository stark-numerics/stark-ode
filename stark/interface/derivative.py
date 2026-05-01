from dataclasses import dataclass, field
from typing import Any, Callable

from stark.carriers.core import CarrierBound
from stark.conventions import Convention, ConventionInPlace, ConventionReturn


@dataclass(slots=True)
class StarkDerivative:
    function: Callable[..., Any]
    convention: Convention = field(default_factory=ConventionReturn)

    def bind(self, carrier: CarrierBound) -> Any:
        if isinstance(self.convention, ConventionReturn):
            return BoundReturnStarkDerivative(self.function, carrier)

        if isinstance(self.convention, ConventionInPlace):
            return BoundInPlaceStarkDerivative(self.function, carrier)

        return BoundStarkDerivative(
            function=self.function,
            convention=self.convention,
            carrier=carrier,
        )

    @classmethod
    def returning(cls, function: Callable[..., Any]) -> "StarkDerivative":
        return cls(function=function, convention=ConventionReturn())

    @classmethod
    def in_place(cls, function: Callable[..., Any]) -> "StarkDerivative":
        return cls(function=function, convention=ConventionInPlace())

    @classmethod
    def from_callable(cls, function: Callable[..., Any]) -> "StarkDerivative":
        return cls.returning(function)


@dataclass(slots=True)
class BoundReturnStarkDerivative:
    function: Callable[..., Any]
    carrier: CarrierBound

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        out.value = self.carrier.coerce_translation(
            self.function(interval.present, state.value)
        )


@dataclass(slots=True)
class BoundInPlaceStarkDerivative:
    function: Callable[..., Any]
    carrier: CarrierBound

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        self.function(interval.present, state.value, out.value)
        self.carrier.validate_translation(out.value)


@dataclass(slots=True)
class BoundStarkDerivative:
    function: Callable[..., Any]
    convention: Convention
    carrier: CarrierBound

    def __call__(self, interval: Any, state: Any, out: Any) -> None:
        out.value = self.convention(
            self.function,
            interval.present,
            state.value,
            out.value,
            self.carrier,
        )