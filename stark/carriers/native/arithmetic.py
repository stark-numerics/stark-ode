from dataclasses import dataclass
from numbers import Number
from typing import Literal

from stark.carriers.native.storage import CarrierNativeValue


@dataclass(frozen=True)
class CarrierArithmeticNative:
    preference: Literal["return"] = "return"

    def translate(self, state: CarrierNativeValue, step: float, derivative: CarrierNativeValue, result: CarrierNativeValue) -> CarrierNativeValue:
        if isinstance(state, Number) and isinstance(derivative, Number):
            return state + step * derivative
        return type(state)(s + step * d for s, d in zip(state, derivative))

    def add(self, left: CarrierNativeValue, right: CarrierNativeValue, result: CarrierNativeValue) -> CarrierNativeValue:
        if isinstance(left, Number) and isinstance(right, Number):
            return left + right
        return type(left)(l + r for l, r in zip(left, right))

    def scale(self, factor: float, value: CarrierNativeValue, result: CarrierNativeValue) -> CarrierNativeValue:
        if isinstance(value, Number):
            return factor * value
        return type(value)(factor * item for item in value)

    def combine(self, terms: tuple[tuple[float, CarrierNativeValue], ...], result: CarrierNativeValue) -> CarrierNativeValue:
        if not terms:
            return result

        first = terms[0][1]

        if isinstance(first, Number):
            return sum(factor * value for factor, value in terms)

        return type(first)(
            sum(factor * value[index] for factor, value in terms)
            for index in range(len(first))
        )