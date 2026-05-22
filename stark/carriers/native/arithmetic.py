from __future__ import annotations

from array import array
from dataclasses import dataclass
from numbers import Number
from typing import Literal

from stark.carriers.native.array import CarrierArithmeticNativeArray
from stark.carriers.native.list import CarrierArithmeticNativeList
from stark.carriers.native.scalar import CarrierArithmeticNativeScalar
from stark.carriers.native.storage import CarrierNativeValue
from stark.carriers.native.tuple import CarrierArithmeticNativeTuple


@dataclass(frozen=True)
class CarrierArithmeticNative:
    """Compatibility dynamic native arithmetic.

    CarrierNative itself does not use this class. It binds one of the concrete
    scalar/list/tuple/array arithmetic implementations during construction so
    tight loops avoid scalar-vs-container branches.
    """

    scalar: CarrierArithmeticNativeScalar = CarrierArithmeticNativeScalar()
    list_: CarrierArithmeticNativeList = CarrierArithmeticNativeList()
    tuple_: CarrierArithmeticNativeTuple = CarrierArithmeticNativeTuple()
    array_: CarrierArithmeticNativeArray = CarrierArithmeticNativeArray()

    @property
    def preference(self) -> Literal["return"]:
        return "return"

    def _select(self, value: CarrierNativeValue):
        if isinstance(value, Number):
            return self.scalar
        if isinstance(value, list):
            return self.list_
        if isinstance(value, tuple):
            return self.tuple_
        if isinstance(value, array):
            return self.array_
        raise TypeError("Native arithmetic value must be numeric, list, tuple, or array.array.")

    def _call(self, name: str, reference: CarrierNativeValue, *args):
        implementation = self._select(reference)
        return getattr(implementation, name)(*args)

    def translate(self, state, step, derivative, result):
        return self._call("translate", state, state, step, derivative, result)

    def add(self, left, right, result):
        return self._call("add", left, left, right, result)

    def scale(self, factor, value, result):
        return self._call("scale", value, factor, value, result)

    def combine2(self, a0, x0, a1, x1, result):
        return self._call("combine2", x0, a0, x0, a1, x1, result)

    def combine3(self, a0, x0, a1, x1, a2, x2, result):
        return self._call("combine3", x0, a0, x0, a1, x1, a2, x2, result)

    def combine4(self, a0, x0, a1, x1, a2, x2, a3, x3, result):
        return self._call("combine4", x0, a0, x0, a1, x1, a2, x2, a3, x3, result)

    def combine5(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, result):
        return self._call("combine5", x0, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, result)

    def combine6(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, result):
        return self._call("combine6", x0, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, result)

    def combine7(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, result):
        return self._call("combine7", x0, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, result)

    def combine8(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, result):
        return self._call("combine8", x0, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, result)

    def combine9(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, result):
        return self._call("combine9", x0, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, result)

    def combine10(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, result):
        return self._call("combine10", x0, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, result)

    def combine11(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, a10, x10, result):
        return self._call("combine11", x0, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, a10, x10, result)

    def combine12(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, a10, x10, a11, x11, result):
        return self._call("combine12", x0, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, a10, x10, a11, x11, result)
