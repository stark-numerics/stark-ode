from dataclasses import dataclass
from typing import Literal, Protocol, cast

import cupy as cp

from stark.carriers.cupy.storage import CarrierCupyValue


class CupyModule(Protocol):
    def multiply(self, left: CarrierCupyValue, right: float, *, out: CarrierCupyValue) -> None: ...
    def add(self, left: CarrierCupyValue, right: CarrierCupyValue, *, out: CarrierCupyValue) -> None: ...


cupy = cast(CupyModule, cp)


@dataclass(frozen=True)
class CarrierArithmeticCupy:
    preference: Literal["into"] = "into"

    def translate(self, state: CarrierCupyValue, step: float, derivative: CarrierCupyValue, result: CarrierCupyValue) -> None:
        cupy.multiply(derivative, step, out=result)
        cupy.add(state, result, out=result)

    def add(self, left: CarrierCupyValue, right: CarrierCupyValue, result: CarrierCupyValue) -> None:
        cupy.add(left, right, out=result)

    def scale(self, factor: float, value: CarrierCupyValue, result: CarrierCupyValue) -> None:
        cupy.multiply(value, factor, out=result)

    def combine(self, terms: tuple[tuple[float, CarrierCupyValue], ...], result: CarrierCupyValue) -> None:
        result[...] = 0
        for factor, value in terms:
            result[...] += factor * value