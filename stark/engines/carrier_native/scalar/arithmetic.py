from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from stark.engines.carrier_native.scalar.storage import CarrierNativeScalarValue


@dataclass(frozen=True)
class CarrierArithmeticNativeScalar:
    @property
    def preference(self) -> Literal["return"]:
        return "return"

    def translate(
        self,
        state: CarrierNativeScalarValue,
        step: float,
        dynamics: CarrierNativeScalarValue,
        result: CarrierNativeScalarValue,
    ) -> CarrierNativeScalarValue:
        return state + step * dynamics

    def add(
        self,
        left: CarrierNativeScalarValue,
        right: CarrierNativeScalarValue,
        result: CarrierNativeScalarValue,
    ) -> CarrierNativeScalarValue:
        return left + right

    def scale(
        self,
        factor: float,
        value: CarrierNativeScalarValue,
        result: CarrierNativeScalarValue,
    ) -> CarrierNativeScalarValue:
        return factor * value

    def combine2(self, a0, x0, a1, x1, result):
        return a0 * x0 + a1 * x1

    def combine3(self, a0, x0, a1, x1, a2, x2, result):
        return a0 * x0 + a1 * x1 + a2 * x2

    def combine4(self, a0, x0, a1, x1, a2, x2, a3, x3, result):
        return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3

    def combine5(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, result):
        return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4

    def combine6(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, result):
        return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5

    def combine7(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, result):
        return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6

    def combine8(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, result):
        return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6 + a7 * x7

    def combine9(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, result):
        return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6 + a7 * x7 + a8 * x8

    def combine10(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, result):
        return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6 + a7 * x7 + a8 * x8 + a9 * x9

    def combine11(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, a10, x10, result):
        return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6 + a7 * x7 + a8 * x8 + a9 * x9 + a10 * x10

    def combine12(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, a10, x10, a11, x11, result):
        return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6 + a7 * x7 + a8 * x8 + a9 * x9 + a10 * x10 + a11 * x11
