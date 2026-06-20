"""In-place arithmetic for CuPy-backed carrier values."""

from typing import Literal, Protocol, cast

import cupy as cp

from stark.engines.cupy.carriers.storage import CarrierCupyValue, CarrierStorageCupy


class HintCupyModule(Protocol):
    """Subset of CuPy elementwise APIs used by carrier arithmetic."""

    def empty(self, shape: tuple[int, ...], dtype: object) -> CarrierCupyValue: ...
    def multiply(self, left: CarrierCupyValue, right: float, *, out: CarrierCupyValue) -> object: ...
    def add(self, left: CarrierCupyValue, right: CarrierCupyValue, *, out: CarrierCupyValue) -> object: ...


cupy = cast(HintCupyModule, cp)


class CarrierArithmeticCupy:
    """Perform carrier algebra by mutating CuPy output arrays in place."""

    storage: CarrierStorageCupy
    scratch: CarrierCupyValue

    def __init__(self, storage: CarrierStorageCupy) -> None:
        self.storage = storage
        self.scratch = cupy.empty(storage.shape, dtype=storage.dtype)

    @property
    def preference(self) -> Literal["into"]:
        return "into"

    def translate(
        self,
        state: CarrierCupyValue,
        step: float,
        derivative: CarrierCupyValue,
        result: CarrierCupyValue,
    ) -> None:
        cupy.multiply(derivative, step, out=self.scratch)
        cupy.add(state, self.scratch, out=result)

    def add(
        self,
        left: CarrierCupyValue,
        right: CarrierCupyValue,
        result: CarrierCupyValue,
    ) -> None:
        cupy.add(left, right, out=result)

    def scale(
        self,
        factor: float,
        value: CarrierCupyValue,
        result: CarrierCupyValue,
    ) -> None:
        cupy.multiply(value, factor, out=result)

    def add_scaled(
        self,
        result: CarrierCupyValue,
        factor: float,
        value: CarrierCupyValue,
    ) -> None:
        cupy.multiply(value, factor, out=self.scratch)
        cupy.add(result, self.scratch, out=result)

    def combine2(
        self,
        a0: float,
        x0: CarrierCupyValue,
        a1: float,
        x1: CarrierCupyValue,
        result: CarrierCupyValue,
    ) -> None:
        cupy.multiply(x0, a0, out=result)
        self.add_scaled(result, a1, x1)

    def combine3(
        self,
        a0: float,
        x0: CarrierCupyValue,
        a1: float,
        x1: CarrierCupyValue,
        a2: float,
        x2: CarrierCupyValue,
        result: CarrierCupyValue,
    ) -> None:
        cupy.multiply(x0, a0, out=result)
        self.add_scaled(result, a1, x1)
        self.add_scaled(result, a2, x2)

    def combine4(
        self,
        a0: float,
        x0: CarrierCupyValue,
        a1: float,
        x1: CarrierCupyValue,
        a2: float,
        x2: CarrierCupyValue,
        a3: float,
        x3: CarrierCupyValue,
        result: CarrierCupyValue,
    ) -> None:
        cupy.multiply(x0, a0, out=result)
        self.add_scaled(result, a1, x1)
        self.add_scaled(result, a2, x2)
        self.add_scaled(result, a3, x3)

    def combine5(
        self,
        a0: float,
        x0: CarrierCupyValue,
        a1: float,
        x1: CarrierCupyValue,
        a2: float,
        x2: CarrierCupyValue,
        a3: float,
        x3: CarrierCupyValue,
        a4: float,
        x4: CarrierCupyValue,
        result: CarrierCupyValue,
    ) -> None:
        cupy.multiply(x0, a0, out=result)
        self.add_scaled(result, a1, x1)
        self.add_scaled(result, a2, x2)
        self.add_scaled(result, a3, x3)
        self.add_scaled(result, a4, x4)

    def combine6(
        self,
        a0: float,
        x0: CarrierCupyValue,
        a1: float,
        x1: CarrierCupyValue,
        a2: float,
        x2: CarrierCupyValue,
        a3: float,
        x3: CarrierCupyValue,
        a4: float,
        x4: CarrierCupyValue,
        a5: float,
        x5: CarrierCupyValue,
        result: CarrierCupyValue,
    ) -> None:
        cupy.multiply(x0, a0, out=result)
        self.add_scaled(result, a1, x1)
        self.add_scaled(result, a2, x2)
        self.add_scaled(result, a3, x3)
        self.add_scaled(result, a4, x4)
        self.add_scaled(result, a5, x5)

    def combine7(
        self,
        a0: float,
        x0: CarrierCupyValue,
        a1: float,
        x1: CarrierCupyValue,
        a2: float,
        x2: CarrierCupyValue,
        a3: float,
        x3: CarrierCupyValue,
        a4: float,
        x4: CarrierCupyValue,
        a5: float,
        x5: CarrierCupyValue,
        a6: float,
        x6: CarrierCupyValue,
        result: CarrierCupyValue,
    ) -> None:
        cupy.multiply(x0, a0, out=result)
        self.add_scaled(result, a1, x1)
        self.add_scaled(result, a2, x2)
        self.add_scaled(result, a3, x3)
        self.add_scaled(result, a4, x4)
        self.add_scaled(result, a5, x5)
        self.add_scaled(result, a6, x6)

    def combine8(
        self,
        a0: float,
        x0: CarrierCupyValue,
        a1: float,
        x1: CarrierCupyValue,
        a2: float,
        x2: CarrierCupyValue,
        a3: float,
        x3: CarrierCupyValue,
        a4: float,
        x4: CarrierCupyValue,
        a5: float,
        x5: CarrierCupyValue,
        a6: float,
        x6: CarrierCupyValue,
        a7: float,
        x7: CarrierCupyValue,
        result: CarrierCupyValue,
    ) -> None:
        cupy.multiply(x0, a0, out=result)
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
        x0: CarrierCupyValue,
        a1: float,
        x1: CarrierCupyValue,
        a2: float,
        x2: CarrierCupyValue,
        a3: float,
        x3: CarrierCupyValue,
        a4: float,
        x4: CarrierCupyValue,
        a5: float,
        x5: CarrierCupyValue,
        a6: float,
        x6: CarrierCupyValue,
        a7: float,
        x7: CarrierCupyValue,
        a8: float,
        x8: CarrierCupyValue,
        result: CarrierCupyValue,
    ) -> None:
        cupy.multiply(x0, a0, out=result)
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
        x0: CarrierCupyValue,
        a1: float,
        x1: CarrierCupyValue,
        a2: float,
        x2: CarrierCupyValue,
        a3: float,
        x3: CarrierCupyValue,
        a4: float,
        x4: CarrierCupyValue,
        a5: float,
        x5: CarrierCupyValue,
        a6: float,
        x6: CarrierCupyValue,
        a7: float,
        x7: CarrierCupyValue,
        a8: float,
        x8: CarrierCupyValue,
        a9: float,
        x9: CarrierCupyValue,
        result: CarrierCupyValue,
    ) -> None:
        cupy.multiply(x0, a0, out=result)
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
        x0: CarrierCupyValue,
        a1: float,
        x1: CarrierCupyValue,
        a2: float,
        x2: CarrierCupyValue,
        a3: float,
        x3: CarrierCupyValue,
        a4: float,
        x4: CarrierCupyValue,
        a5: float,
        x5: CarrierCupyValue,
        a6: float,
        x6: CarrierCupyValue,
        a7: float,
        x7: CarrierCupyValue,
        a8: float,
        x8: CarrierCupyValue,
        a9: float,
        x9: CarrierCupyValue,
        a10: float,
        x10: CarrierCupyValue,
        result: CarrierCupyValue,
    ) -> None:
        cupy.multiply(x0, a0, out=result)
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
        x0: CarrierCupyValue,
        a1: float,
        x1: CarrierCupyValue,
        a2: float,
        x2: CarrierCupyValue,
        a3: float,
        x3: CarrierCupyValue,
        a4: float,
        x4: CarrierCupyValue,
        a5: float,
        x5: CarrierCupyValue,
        a6: float,
        x6: CarrierCupyValue,
        a7: float,
        x7: CarrierCupyValue,
        a8: float,
        x8: CarrierCupyValue,
        a9: float,
        x9: CarrierCupyValue,
        a10: float,
        x10: CarrierCupyValue,
        a11: float,
        x11: CarrierCupyValue,
        result: CarrierCupyValue,
    ) -> None:
        cupy.multiply(x0, a0, out=result)
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
