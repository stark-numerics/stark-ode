from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from stark.engines.carriers.native.list.storage import CarrierNativeListValue


@dataclass(frozen=True)
class CarrierArithmeticNativeList:
    @property
    def preference(self) -> Literal["return"]:
        return "return"

    def translate(
        self,
        state: CarrierNativeListValue,
        step: float,
        derivative: CarrierNativeListValue,
        result: CarrierNativeListValue,
    ) -> CarrierNativeListValue:
        return [item + step * delta for item, delta in zip(state, derivative)]

    def add(
        self,
        left: CarrierNativeListValue,
        right: CarrierNativeListValue,
        result: CarrierNativeListValue,
    ) -> CarrierNativeListValue:
        return [left_item + right_item for left_item, right_item in zip(left, right)]

    def scale(
        self,
        factor: float,
        value: CarrierNativeListValue,
        result: CarrierNativeListValue,
    ) -> CarrierNativeListValue:
        return [factor * item for item in value]

    def _combine(self, coefficients: tuple[float, ...], values: tuple[CarrierNativeListValue, ...]) -> CarrierNativeListValue:
        length = len(values[0])
        return [
            sum(coefficient * value[index] for coefficient, value in zip(coefficients, values))
            for index in range(length)
        ]

    def combine2(self, a0, x0, a1, x1, result):
        return self._combine((a0, a1), (x0, x1))

    def combine3(self, a0, x0, a1, x1, a2, x2, result):
        return self._combine((a0, a1, a2), (x0, x1, x2))

    def combine4(self, a0, x0, a1, x1, a2, x2, a3, x3, result):
        return self._combine((a0, a1, a2, a3), (x0, x1, x2, x3))

    def combine5(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, result):
        return self._combine((a0, a1, a2, a3, a4), (x0, x1, x2, x3, x4))

    def combine6(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, result):
        return self._combine((a0, a1, a2, a3, a4, a5), (x0, x1, x2, x3, x4, x5))

    def combine7(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, result):
        return self._combine((a0, a1, a2, a3, a4, a5, a6), (x0, x1, x2, x3, x4, x5, x6))

    def combine8(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, result):
        return self._combine((a0, a1, a2, a3, a4, a5, a6, a7), (x0, x1, x2, x3, x4, x5, x6, x7))

    def combine9(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, result):
        return self._combine((a0, a1, a2, a3, a4, a5, a6, a7, a8), (x0, x1, x2, x3, x4, x5, x6, x7, x8))

    def combine10(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, result):
        return self._combine((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9), (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9))

    def combine11(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, a10, x10, result):
        return self._combine((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10), (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10))

    def combine12(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, a10, x10, a11, x11, result):
        return self._combine((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11), (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11))
