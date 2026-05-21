from numbers import Number
from typing import Literal

from stark.carriers.native.storage import CarrierNativeValue


class CarrierArithmeticNative:
    @property
    def preference(self) -> Literal["return"]:
        return "return"

    def translate(
        self,
        state: CarrierNativeValue,
        step: float,
        derivative: CarrierNativeValue,
        result: CarrierNativeValue,
    ) -> CarrierNativeValue:
        if isinstance(state, Number):
            return state + step * derivative

        return type(state)(
            item + step * delta
            for item, delta in zip(state, derivative)
        )

    def add(
        self,
        left: CarrierNativeValue,
        right: CarrierNativeValue,
        result: CarrierNativeValue,
    ) -> CarrierNativeValue:
        if isinstance(left, Number):
            return left + right

        return type(left)(
            left_item + right_item
            for left_item, right_item in zip(left, right)
        )

    def scale(
        self,
        factor: float,
        value: CarrierNativeValue,
        result: CarrierNativeValue,
    ) -> CarrierNativeValue:
        if isinstance(value, Number):
            return factor * value

        return type(value)(factor * item for item in value)

    def combine2(
        self,
        a0: float,
        x0: CarrierNativeValue,
        a1: float,
        x1: CarrierNativeValue,
        result: CarrierNativeValue,
    ) -> CarrierNativeValue:
        if isinstance(x0, Number):
            return a0 * x0 + a1 * x1

        return type(x0)(
            a0 * x0[index] + a1 * x1[index]
            for index in range(len(x0))
        )

    def combine3(
        self,
        a0: float,
        x0: CarrierNativeValue,
        a1: float,
        x1: CarrierNativeValue,
        a2: float,
        x2: CarrierNativeValue,
        result: CarrierNativeValue,
    ) -> CarrierNativeValue:
        if isinstance(x0, Number):
            return a0 * x0 + a1 * x1 + a2 * x2

        return type(x0)(
            a0 * x0[index] + a1 * x1[index] + a2 * x2[index]
            for index in range(len(x0))
        )

    def combine4(
        self,
        a0: float,
        x0: CarrierNativeValue,
        a1: float,
        x1: CarrierNativeValue,
        a2: float,
        x2: CarrierNativeValue,
        a3: float,
        x3: CarrierNativeValue,
        result: CarrierNativeValue,
    ) -> CarrierNativeValue:
        if isinstance(x0, Number):
            return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3

        return type(x0)(
            a0 * x0[index] + a1 * x1[index] + a2 * x2[index] + a3 * x3[index]
            for index in range(len(x0))
        )

    def combine5(
        self,
        a0: float,
        x0: CarrierNativeValue,
        a1: float,
        x1: CarrierNativeValue,
        a2: float,
        x2: CarrierNativeValue,
        a3: float,
        x3: CarrierNativeValue,
        a4: float,
        x4: CarrierNativeValue,
        result: CarrierNativeValue,
    ) -> CarrierNativeValue:
        if isinstance(x0, Number):
            return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4

        return type(x0)(
            a0 * x0[index] + a1 * x1[index] + a2 * x2[index] + a3 * x3[index] + a4 * x4[index]
            for index in range(len(x0))
        )

    def combine6(
        self,
        a0: float,
        x0: CarrierNativeValue,
        a1: float,
        x1: CarrierNativeValue,
        a2: float,
        x2: CarrierNativeValue,
        a3: float,
        x3: CarrierNativeValue,
        a4: float,
        x4: CarrierNativeValue,
        a5: float,
        x5: CarrierNativeValue,
        result: CarrierNativeValue,
    ) -> CarrierNativeValue:
        if isinstance(x0, Number):
            return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5

        return type(x0)(
            a0 * x0[index] + a1 * x1[index] + a2 * x2[index] + a3 * x3[index] + a4 * x4[index] + a5 * x5[index]
            for index in range(len(x0))
        )

    def combine7(
        self,
        a0: float,
        x0: CarrierNativeValue,
        a1: float,
        x1: CarrierNativeValue,
        a2: float,
        x2: CarrierNativeValue,
        a3: float,
        x3: CarrierNativeValue,
        a4: float,
        x4: CarrierNativeValue,
        a5: float,
        x5: CarrierNativeValue,
        a6: float,
        x6: CarrierNativeValue,
        result: CarrierNativeValue,
    ) -> CarrierNativeValue:
        if isinstance(x0, Number):
            return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6

        return type(x0)(
            a0 * x0[index] + a1 * x1[index] + a2 * x2[index] + a3 * x3[index] + a4 * x4[index] + a5 * x5[index] + a6 * x6[index]
            for index in range(len(x0))
        )

    def combine8(
        self,
        a0: float,
        x0: CarrierNativeValue,
        a1: float,
        x1: CarrierNativeValue,
        a2: float,
        x2: CarrierNativeValue,
        a3: float,
        x3: CarrierNativeValue,
        a4: float,
        x4: CarrierNativeValue,
        a5: float,
        x5: CarrierNativeValue,
        a6: float,
        x6: CarrierNativeValue,
        a7: float,
        x7: CarrierNativeValue,
        result: CarrierNativeValue,
    ) -> CarrierNativeValue:
        if isinstance(x0, Number):
            return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6 + a7 * x7

        return type(x0)(
            a0 * x0[index] + a1 * x1[index] + a2 * x2[index] + a3 * x3[index] + a4 * x4[index] + a5 * x5[index] + a6 * x6[index] + a7 * x7[index]
            for index in range(len(x0))
        )

    def combine9(
        self,
        a0: float,
        x0: CarrierNativeValue,
        a1: float,
        x1: CarrierNativeValue,
        a2: float,
        x2: CarrierNativeValue,
        a3: float,
        x3: CarrierNativeValue,
        a4: float,
        x4: CarrierNativeValue,
        a5: float,
        x5: CarrierNativeValue,
        a6: float,
        x6: CarrierNativeValue,
        a7: float,
        x7: CarrierNativeValue,
        a8: float,
        x8: CarrierNativeValue,
        result: CarrierNativeValue,
    ) -> CarrierNativeValue:
        if isinstance(x0, Number):
            return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6 + a7 * x7 + a8 * x8

        return type(x0)(
            a0 * x0[index] + a1 * x1[index] + a2 * x2[index] + a3 * x3[index] + a4 * x4[index] + a5 * x5[index] + a6 * x6[index] + a7 * x7[index] + a8 * x8[index]
            for index in range(len(x0))
        )

    def combine10(
        self,
        a0: float,
        x0: CarrierNativeValue,
        a1: float,
        x1: CarrierNativeValue,
        a2: float,
        x2: CarrierNativeValue,
        a3: float,
        x3: CarrierNativeValue,
        a4: float,
        x4: CarrierNativeValue,
        a5: float,
        x5: CarrierNativeValue,
        a6: float,
        x6: CarrierNativeValue,
        a7: float,
        x7: CarrierNativeValue,
        a8: float,
        x8: CarrierNativeValue,
        a9: float,
        x9: CarrierNativeValue,
        result: CarrierNativeValue,
    ) -> CarrierNativeValue:
        if isinstance(x0, Number):
            return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6 + a7 * x7 + a8 * x8 + a9 * x9

        return type(x0)(
            a0 * x0[index] + a1 * x1[index] + a2 * x2[index] + a3 * x3[index] + a4 * x4[index] + a5 * x5[index] + a6 * x6[index] + a7 * x7[index] + a8 * x8[index] + a9 * x9[index]
            for index in range(len(x0))
        )

    def combine11(
        self,
        a0: float,
        x0: CarrierNativeValue,
        a1: float,
        x1: CarrierNativeValue,
        a2: float,
        x2: CarrierNativeValue,
        a3: float,
        x3: CarrierNativeValue,
        a4: float,
        x4: CarrierNativeValue,
        a5: float,
        x5: CarrierNativeValue,
        a6: float,
        x6: CarrierNativeValue,
        a7: float,
        x7: CarrierNativeValue,
        a8: float,
        x8: CarrierNativeValue,
        a9: float,
        x9: CarrierNativeValue,
        a10: float,
        x10: CarrierNativeValue,
        result: CarrierNativeValue,
    ) -> CarrierNativeValue:
        if isinstance(x0, Number):
            return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6 + a7 * x7 + a8 * x8 + a9 * x9 + a10 * x10

        return type(x0)(
            a0 * x0[index] + a1 * x1[index] + a2 * x2[index] + a3 * x3[index] + a4 * x4[index] + a5 * x5[index] + a6 * x6[index] + a7 * x7[index] + a8 * x8[index] + a9 * x9[index] + a10 * x10[index]
            for index in range(len(x0))
        )

    def combine12(
        self,
        a0: float,
        x0: CarrierNativeValue,
        a1: float,
        x1: CarrierNativeValue,
        a2: float,
        x2: CarrierNativeValue,
        a3: float,
        x3: CarrierNativeValue,
        a4: float,
        x4: CarrierNativeValue,
        a5: float,
        x5: CarrierNativeValue,
        a6: float,
        x6: CarrierNativeValue,
        a7: float,
        x7: CarrierNativeValue,
        a8: float,
        x8: CarrierNativeValue,
        a9: float,
        x9: CarrierNativeValue,
        a10: float,
        x10: CarrierNativeValue,
        a11: float,
        x11: CarrierNativeValue,
        result: CarrierNativeValue,
    ) -> CarrierNativeValue:
        if isinstance(x0, Number):
            return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6 + a7 * x7 + a8 * x8 + a9 * x9 + a10 * x10 + a11 * x11

        return type(x0)(
            a0 * x0[index] + a1 * x1[index] + a2 * x2[index] + a3 * x3[index] + a4 * x4[index] + a5 * x5[index] + a6 * x6[index] + a7 * x7[index] + a8 * x8[index] + a9 * x9[index] + a10 * x10[index] + a11 * x11[index]
            for index in range(len(x0))
        )

