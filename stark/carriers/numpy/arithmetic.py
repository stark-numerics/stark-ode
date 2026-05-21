from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from stark.carriers.numpy.storage import CarrierStorageNumpy

Array = NDArray[Any]


class CarrierArithmeticNumpy:
    storage: CarrierStorageNumpy
    scratch: Array

    def __init__(self, storage: CarrierStorageNumpy) -> None:
        self.storage = storage
        self.scratch = np.empty(storage.shape, dtype=storage.dtype)
    
    @property
    def preference(self) -> Literal["into"]:
        return "into"

    def translate(
        self,
        state: Array,
        step: float,
        derivative: Array,
        result: Array,
    ) -> None:
        np.multiply(derivative, step, out=result)
        np.add(state, result, out=result)

    def add(self, left: Array, right: Array, result: Array) -> None:
        np.add(left, right, out=result)

    def scale(self, factor: float, value: Array, result: Array) -> None:
        np.multiply(value, factor, out=result)

    def add_scaled(self, result: Array, factor: float, value: Array) -> None:
        np.multiply(value, factor, out=self.scratch)
        np.add(result, self.scratch, out=result)

    def combine2(
        self,
        a0: float,
        x0: Array,
        a1: float,
        x1: Array,
        result: Array,
    ) -> None:
        np.multiply(x0, a0, out=result)
        self.add_scaled(result, a1, x1)

    def combine3(
        self,
        a0: float,
        x0: Array,
        a1: float,
        x1: Array,
        a2: float,
        x2: Array,
        result: Array,
    ) -> None:
        np.multiply(x0, a0, out=result)
        self.add_scaled(result, a1, x1)
        self.add_scaled(result, a2, x2)

    def combine4(
        self,
        a0: float,
        x0: Array,
        a1: float,
        x1: Array,
        a2: float,
        x2: Array,
        a3: float,
        x3: Array,
        result: Array,
    ) -> None:
        np.multiply(x0, a0, out=result)
        self.add_scaled(result, a1, x1)
        self.add_scaled(result, a2, x2)
        self.add_scaled(result, a3, x3)

    def combine5(
        self,
        a0: float,
        x0: Array,
        a1: float,
        x1: Array,
        a2: float,
        x2: Array,
        a3: float,
        x3: Array,
        a4: float,
        x4: Array,
        result: Array,
    ) -> None:
        np.multiply(x0, a0, out=result)
        self.add_scaled(result, a1, x1)
        self.add_scaled(result, a2, x2)
        self.add_scaled(result, a3, x3)
        self.add_scaled(result, a4, x4)

    def combine6(
        self,
        a0: float,
        x0: Array,
        a1: float,
        x1: Array,
        a2: float,
        x2: Array,
        a3: float,
        x3: Array,
        a4: float,
        x4: Array,
        a5: float,
        x5: Array,
        result: Array,
    ) -> None:
        np.multiply(x0, a0, out=result)
        self.add_scaled(result, a1, x1)
        self.add_scaled(result, a2, x2)
        self.add_scaled(result, a3, x3)
        self.add_scaled(result, a4, x4)
        self.add_scaled(result, a5, x5)

    def combine7(
        self,
        a0: float,
        x0: Array,
        a1: float,
        x1: Array,
        a2: float,
        x2: Array,
        a3: float,
        x3: Array,
        a4: float,
        x4: Array,
        a5: float,
        x5: Array,
        a6: float,
        x6: Array,
        result: Array,
    ) -> None:
        np.multiply(x0, a0, out=result)
        self.add_scaled(result, a1, x1)
        self.add_scaled(result, a2, x2)
        self.add_scaled(result, a3, x3)
        self.add_scaled(result, a4, x4)
        self.add_scaled(result, a5, x5)
        self.add_scaled(result, a6, x6)

    def combine8(
        self,
        a0: float,
        x0: Array,
        a1: float,
        x1: Array,
        a2: float,
        x2: Array,
        a3: float,
        x3: Array,
        a4: float,
        x4: Array,
        a5: float,
        x5: Array,
        a6: float,
        x6: Array,
        a7: float,
        x7: Array,
        result: Array,
    ) -> None:
        np.multiply(x0, a0, out=result)
        self.add_scaled(result, a1, x1)
        self.add_scaled(result, a2, x2)
        self.add_scaled(result, a3, x3)
        self.add_scaled(result, a4, x4)
        self.add_scaled(result, a5, x5)
        self.add_scaled(result, a6, x6)
        self.add_scaled(result, a7, x7)

    def combine9(
        self,
        a0: float,
        x0: Array,
        a1: float,
        x1: Array,
        a2: float,
        x2: Array,
        a3: float,
        x3: Array,
        a4: float,
        x4: Array,
        a5: float,
        x5: Array,
        a6: float,
        x6: Array,
        a7: float,
        x7: Array,
        a8: float,
        x8: Array,
        result: Array,
    ) -> None:
        np.multiply(x0, a0, out=result)
        self.add_scaled(result, a1, x1)
        self.add_scaled(result, a2, x2)
        self.add_scaled(result, a3, x3)
        self.add_scaled(result, a4, x4)
        self.add_scaled(result, a5, x5)
        self.add_scaled(result, a6, x6)
        self.add_scaled(result, a7, x7)
        self.add_scaled(result, a8, x8)

    def combine10(
        self,
        a0: float,
        x0: Array,
        a1: float,
        x1: Array,
        a2: float,
        x2: Array,
        a3: float,
        x3: Array,
        a4: float,
        x4: Array,
        a5: float,
        x5: Array,
        a6: float,
        x6: Array,
        a7: float,
        x7: Array,
        a8: float,
        x8: Array,
        a9: float,
        x9: Array,
        result: Array,
    ) -> None:
        np.multiply(x0, a0, out=result)
        self.add_scaled(result, a1, x1)
        self.add_scaled(result, a2, x2)
        self.add_scaled(result, a3, x3)
        self.add_scaled(result, a4, x4)
        self.add_scaled(result, a5, x5)
        self.add_scaled(result, a6, x6)
        self.add_scaled(result, a7, x7)
        self.add_scaled(result, a8, x8)
        self.add_scaled(result, a9, x9)

    def combine11(
        self,
        a0: float,
        x0: Array,
        a1: float,
        x1: Array,
        a2: float,
        x2: Array,
        a3: float,
        x3: Array,
        a4: float,
        x4: Array,
        a5: float,
        x5: Array,
        a6: float,
        x6: Array,
        a7: float,
        x7: Array,
        a8: float,
        x8: Array,
        a9: float,
        x9: Array,
        a10: float,
        x10: Array,
        result: Array,
    ) -> None:
        np.multiply(x0, a0, out=result)
        self.add_scaled(result, a1, x1)
        self.add_scaled(result, a2, x2)
        self.add_scaled(result, a3, x3)
        self.add_scaled(result, a4, x4)
        self.add_scaled(result, a5, x5)
        self.add_scaled(result, a6, x6)
        self.add_scaled(result, a7, x7)
        self.add_scaled(result, a8, x8)
        self.add_scaled(result, a9, x9)
        self.add_scaled(result, a10, x10)

    def combine12(
        self,
        a0: float,
        x0: Array,
        a1: float,
        x1: Array,
        a2: float,
        x2: Array,
        a3: float,
        x3: Array,
        a4: float,
        x4: Array,
        a5: float,
        x5: Array,
        a6: float,
        x6: Array,
        a7: float,
        x7: Array,
        a8: float,
        x8: Array,
        a9: float,
        x9: Array,
        a10: float,
        x10: Array,
        a11: float,
        x11: Array,
        result: Array,
    ) -> None:
        np.multiply(x0, a0, out=result)
        self.add_scaled(result, a1, x1)
        self.add_scaled(result, a2, x2)
        self.add_scaled(result, a3, x3)
        self.add_scaled(result, a4, x4)
        self.add_scaled(result, a5, x5)
        self.add_scaled(result, a6, x6)
        self.add_scaled(result, a7, x7)
        self.add_scaled(result, a8, x8)
        self.add_scaled(result, a9, x9)
        self.add_scaled(result, a10, x10)
        self.add_scaled(result, a11, x11)

