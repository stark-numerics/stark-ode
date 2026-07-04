from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from stark.engines.native.carriers.array.storage import CarrierNativeArrayValue


@dataclass(frozen=True)
class CarrierArithmeticNativeArray:
    @property
    def preference(self) -> Literal["into"]:
        return "into"

    def translate(
        self,
        state: CarrierNativeArrayValue,
        step: float,
        dynamics: CarrierNativeArrayValue,
        result: CarrierNativeArrayValue,
    ) -> None:
        for index in range(len(state)):
            result[index] = state[index] + step * dynamics[index]

    def add(
        self,
        left: CarrierNativeArrayValue,
        right: CarrierNativeArrayValue,
        result: CarrierNativeArrayValue,
    ) -> None:
        for index in range(len(left)):
            result[index] = left[index] + right[index]

    def scale(
        self,
        factor: float,
        value: CarrierNativeArrayValue,
        result: CarrierNativeArrayValue,
    ) -> None:
        for index in range(len(value)):
            result[index] = factor * value[index]

    def _combine_into(
        self,
        coefficients: tuple[float, ...],
        values: tuple[CarrierNativeArrayValue, ...],
        result: CarrierNativeArrayValue,
    ) -> None:
        for index in range(len(result)):
            result[index] = sum(
                coefficient * value[index]
                for coefficient, value in zip(coefficients, values)
            )

    def combine2(self, a0, x0, a1, x1, result):
        self._combine_into((a0, a1), (x0, x1), result)

    def combine3(self, a0, x0, a1, x1, a2, x2, result):
        self._combine_into((a0, a1, a2), (x0, x1, x2), result)

    def combine4(self, a0, x0, a1, x1, a2, x2, a3, x3, result):
        self._combine_into((a0, a1, a2, a3), (x0, x1, x2, x3), result)

    def combine5(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, result):
        self._combine_into((a0, a1, a2, a3, a4), (x0, x1, x2, x3, x4), result)

    def combine6(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, result):
        self._combine_into((a0, a1, a2, a3, a4, a5), (x0, x1, x2, x3, x4, x5), result)

    def combine7(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, result):
        self._combine_into((a0, a1, a2, a3, a4, a5, a6), (x0, x1, x2, x3, x4, x5, x6), result)

    def combine8(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, result):
        self._combine_into((a0, a1, a2, a3, a4, a5, a6, a7), (x0, x1, x2, x3, x4, x5, x6, x7), result)

    def combine9(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, result):
        self._combine_into((a0, a1, a2, a3, a4, a5, a6, a7, a8), (x0, x1, x2, x3, x4, x5, x6, x7, x8), result)

    def combine10(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, result):
        self._combine_into((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9), (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9), result)

    def combine11(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, a10, x10, result):
        self._combine_into((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10), (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10), result)

    def combine12(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, a10, x10, a11, x11, result):
        self._combine_into((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11), (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11), result)
