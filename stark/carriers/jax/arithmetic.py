from dataclasses import dataclass
from typing import Literal, Protocol, cast

import jax.numpy as jnp  # type: ignore[import-not-found]

from stark.carriers.jax.storage import CarrierJaxValue


class JaxNumpyModule(Protocol):
    def zeros_like(self, value: CarrierJaxValue) -> CarrierJaxValue: ...


jax_numpy = cast(JaxNumpyModule, jnp)


@dataclass(frozen=True)
class CarrierArithmeticJax:
    preference: Literal["return"] = "return"

    def translate(self, state: CarrierJaxValue, step: float, derivative: CarrierJaxValue, result: CarrierJaxValue) -> CarrierJaxValue:
        return state + step * derivative

    def add(self, left: CarrierJaxValue, right: CarrierJaxValue, result: CarrierJaxValue) -> CarrierJaxValue:
        return left + right

    def scale(self, factor: float, value: CarrierJaxValue, result: CarrierJaxValue) -> CarrierJaxValue:
        return factor * value

    def combine(self, terms: tuple[tuple[float, CarrierJaxValue], ...], result: CarrierJaxValue) -> CarrierJaxValue:
        total = jax_numpy.zeros_like(result)
        for factor, value in terms:
            total = total + factor * value
        return total