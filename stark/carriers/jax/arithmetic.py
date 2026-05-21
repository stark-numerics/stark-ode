from typing import Literal

from stark.carriers.jax.storage import CarrierJaxValue


class CarrierArithmeticJax:
    @property
    def preference(self) -> Literal["return"]:
        return "return"

    def translate(
        self,
        state: CarrierJaxValue,
        step: float,
        derivative: CarrierJaxValue,
        result: CarrierJaxValue,
    ) -> CarrierJaxValue:
        return state + step * derivative

    def add(
        self,
        left: CarrierJaxValue,
        right: CarrierJaxValue,
        result: CarrierJaxValue,
    ) -> CarrierJaxValue:
        return left + right

    def scale(
        self,
        factor: float,
        value: CarrierJaxValue,
        result: CarrierJaxValue,
    ) -> CarrierJaxValue:
        return factor * value

    def combine2(
        self,
        a0: float,
        x0: CarrierJaxValue,
        a1: float,
        x1: CarrierJaxValue,
        result: CarrierJaxValue,
    ) -> CarrierJaxValue:
        return a0 * x0 + a1 * x1

    def combine3(
        self,
        a0: float,
        x0: CarrierJaxValue,
        a1: float,
        x1: CarrierJaxValue,
        a2: float,
        x2: CarrierJaxValue,
        result: CarrierJaxValue,
    ) -> CarrierJaxValue:
        return a0 * x0 + a1 * x1 + a2 * x2

    def combine4(
        self,
        a0: float,
        x0: CarrierJaxValue,
        a1: float,
        x1: CarrierJaxValue,
        a2: float,
        x2: CarrierJaxValue,
        a3: float,
        x3: CarrierJaxValue,
        result: CarrierJaxValue,
    ) -> CarrierJaxValue:
        return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3

    def combine5(
        self,
        a0: float,
        x0: CarrierJaxValue,
        a1: float,
        x1: CarrierJaxValue,
        a2: float,
        x2: CarrierJaxValue,
        a3: float,
        x3: CarrierJaxValue,
        a4: float,
        x4: CarrierJaxValue,
        result: CarrierJaxValue,
    ) -> CarrierJaxValue:
        return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4

    def combine6(
        self,
        a0: float,
        x0: CarrierJaxValue,
        a1: float,
        x1: CarrierJaxValue,
        a2: float,
        x2: CarrierJaxValue,
        a3: float,
        x3: CarrierJaxValue,
        a4: float,
        x4: CarrierJaxValue,
        a5: float,
        x5: CarrierJaxValue,
        result: CarrierJaxValue,
    ) -> CarrierJaxValue:
        return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5

    def combine7(
        self,
        a0: float,
        x0: CarrierJaxValue,
        a1: float,
        x1: CarrierJaxValue,
        a2: float,
        x2: CarrierJaxValue,
        a3: float,
        x3: CarrierJaxValue,
        a4: float,
        x4: CarrierJaxValue,
        a5: float,
        x5: CarrierJaxValue,
        a6: float,
        x6: CarrierJaxValue,
        result: CarrierJaxValue,
    ) -> CarrierJaxValue:
        return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6

    def combine8(
        self,
        a0: float,
        x0: CarrierJaxValue,
        a1: float,
        x1: CarrierJaxValue,
        a2: float,
        x2: CarrierJaxValue,
        a3: float,
        x3: CarrierJaxValue,
        a4: float,
        x4: CarrierJaxValue,
        a5: float,
        x5: CarrierJaxValue,
        a6: float,
        x6: CarrierJaxValue,
        a7: float,
        x7: CarrierJaxValue,
        result: CarrierJaxValue,
    ) -> CarrierJaxValue:
        return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6 + a7 * x7

    def combine9(
        self,
        a0: float,
        x0: CarrierJaxValue,
        a1: float,
        x1: CarrierJaxValue,
        a2: float,
        x2: CarrierJaxValue,
        a3: float,
        x3: CarrierJaxValue,
        a4: float,
        x4: CarrierJaxValue,
        a5: float,
        x5: CarrierJaxValue,
        a6: float,
        x6: CarrierJaxValue,
        a7: float,
        x7: CarrierJaxValue,
        a8: float,
        x8: CarrierJaxValue,
        result: CarrierJaxValue,
    ) -> CarrierJaxValue:
        return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6 + a7 * x7 + a8 * x8

    def combine10(
        self,
        a0: float,
        x0: CarrierJaxValue,
        a1: float,
        x1: CarrierJaxValue,
        a2: float,
        x2: CarrierJaxValue,
        a3: float,
        x3: CarrierJaxValue,
        a4: float,
        x4: CarrierJaxValue,
        a5: float,
        x5: CarrierJaxValue,
        a6: float,
        x6: CarrierJaxValue,
        a7: float,
        x7: CarrierJaxValue,
        a8: float,
        x8: CarrierJaxValue,
        a9: float,
        x9: CarrierJaxValue,
        result: CarrierJaxValue,
    ) -> CarrierJaxValue:
        return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6 + a7 * x7 + a8 * x8 + a9 * x9

    def combine11(
        self,
        a0: float,
        x0: CarrierJaxValue,
        a1: float,
        x1: CarrierJaxValue,
        a2: float,
        x2: CarrierJaxValue,
        a3: float,
        x3: CarrierJaxValue,
        a4: float,
        x4: CarrierJaxValue,
        a5: float,
        x5: CarrierJaxValue,
        a6: float,
        x6: CarrierJaxValue,
        a7: float,
        x7: CarrierJaxValue,
        a8: float,
        x8: CarrierJaxValue,
        a9: float,
        x9: CarrierJaxValue,
        a10: float,
        x10: CarrierJaxValue,
        result: CarrierJaxValue,
    ) -> CarrierJaxValue:
        return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6 + a7 * x7 + a8 * x8 + a9 * x9 + a10 * x10

    def combine12(
        self,
        a0: float,
        x0: CarrierJaxValue,
        a1: float,
        x1: CarrierJaxValue,
        a2: float,
        x2: CarrierJaxValue,
        a3: float,
        x3: CarrierJaxValue,
        a4: float,
        x4: CarrierJaxValue,
        a5: float,
        x5: CarrierJaxValue,
        a6: float,
        x6: CarrierJaxValue,
        a7: float,
        x7: CarrierJaxValue,
        a8: float,
        x8: CarrierJaxValue,
        a9: float,
        x9: CarrierJaxValue,
        a10: float,
        x10: CarrierJaxValue,
        a11: float,
        x11: CarrierJaxValue,
        result: CarrierJaxValue,
    ) -> CarrierJaxValue:
        return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6 + a7 * x7 + a8 * x8 + a9 * x9 + a10 * x10 + a11 * x11

