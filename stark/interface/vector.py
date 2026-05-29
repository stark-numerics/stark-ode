from __future__ import annotations

from typing import Any, Callable

from stark.contracts import Carrier


class StarkVector:
    def __init__(self, value: Any, carrier: Carrier[Any, Any]) -> None:
        self.value = value
        self.carrier = carrier

    def translation(self, value: Any) -> "StarkVectorTranslation":
        return StarkVectorTranslation(value, self.carrier)

    def zero_translation(self) -> "StarkVectorTranslation":
        return self.translation(self.carrier.allocation.zero_translation())

    def allocator(self) -> "StarkVectorAllocator":
        return StarkVectorAllocator(self.carrier)


class StarkVectorTranslation:
    apply: Callable[["StarkVector", "StarkVector"], None]
    add: Callable[["StarkVectorTranslation", "StarkVectorTranslation"], "StarkVectorTranslation"]
    scale: Callable[[float, "StarkVectorTranslation", "StarkVectorTranslation"], "StarkVectorTranslation"]

    def __init__(self, value: Any, carrier: Carrier[Any, Any]) -> None:
        self.value = value
        self.carrier = carrier

        if carrier.arithmetic.preference == "into":
            self.apply = self.apply_into
            self.add = self.add_into
            self.scale = self.scale_into

            self.combine2 = self.combine2_into
            self.combine3 = self.combine3_into
            self.combine4 = self.combine4_into
            self.combine5 = self.combine5_into
            self.combine6 = self.combine6_into
            self.combine7 = self.combine7_into
            self.combine8 = self.combine8_into
            self.combine9 = self.combine9_into
            self.combine10 = self.combine10_into
            self.combine11 = self.combine11_into
            self.combine12 = self.combine12_into
        else:
            self.apply = self.apply_return
            self.add = self.add_return
            self.scale = self.scale_return

            self.combine2 = self.combine2_return
            self.combine3 = self.combine3_return
            self.combine4 = self.combine4_return
            self.combine5 = self.combine5_return
            self.combine6 = self.combine6_return
            self.combine7 = self.combine7_return
            self.combine8 = self.combine8_return
            self.combine9 = self.combine9_return
            self.combine10 = self.combine10_return
            self.combine11 = self.combine11_return
            self.combine12 = self.combine12_return

    def __call__(self, origin: StarkVector, result: StarkVector) -> None:
        self.apply(origin, result)

    def norm(self) -> float:
        return self.carrier.norm(self.value)

    @property
    def linear_combine(self) -> tuple[Callable[..., "StarkVectorTranslation"], ...]:
        return (
            self.scale,
            self.combine2,
            self.combine3,
            self.combine4,
            self.combine5,
            self.combine6,
            self.combine7,
            self.combine8,
            self.combine9,
            self.combine10,
            self.combine11,
            self.combine12,
        )

    def apply_into(self, origin: StarkVector, result: StarkVector) -> None:
        self.carrier.arithmetic.translate(origin.value, 1.0, self.value, result.value)

    def apply_return(self, origin: StarkVector, result: StarkVector) -> None:
        result.value = self.carrier.arithmetic.translate(origin.value, 1.0, self.value, result.value)

    def add_into(self, other: "StarkVectorTranslation", result: "StarkVectorTranslation") -> "StarkVectorTranslation":
        self.carrier.arithmetic.add(self.value, other.value, result.value)
        return result

    def add_return(self, other: "StarkVectorTranslation", result: "StarkVectorTranslation") -> "StarkVectorTranslation":
        result.value = self.carrier.arithmetic.add(self.value, other.value, result.value)
        return result

    def scale_into(self, coefficient: float, value: "StarkVectorTranslation", out: "StarkVectorTranslation") -> "StarkVectorTranslation":
        self.carrier.arithmetic.scale(coefficient, value.value, out.value)
        return out

    def scale_return(self, coefficient: float, value: "StarkVectorTranslation", out: "StarkVectorTranslation") -> "StarkVectorTranslation":
        out.value = self.carrier.arithmetic.scale(coefficient, value.value, out.value)
        return out

    def combine2_into(self, a0, x0, a1, x1, out):
        self.carrier.arithmetic.combine2(a0, x0.value, a1, x1.value, out.value)
        return out

    def combine2_return(self, a0, x0, a1, x1, out):
        out.value = self.carrier.arithmetic.combine2(a0, x0.value, a1, x1.value, out.value)
        return out

    def combine3_into(self, a0, x0, a1, x1, a2, x2, out):
        self.carrier.arithmetic.combine3(a0, x0.value, a1, x1.value, a2, x2.value, out.value)
        return out

    def combine3_return(self, a0, x0, a1, x1, a2, x2, out):
        out.value = self.carrier.arithmetic.combine3(a0, x0.value, a1, x1.value, a2, x2.value, out.value)
        return out

    def combine4_into(self, a0, x0, a1, x1, a2, x2, a3, x3, out):
        self.carrier.arithmetic.combine4(a0, x0.value, a1, x1.value, a2, x2.value, a3, x3.value, out.value)
        return out

    def combine4_return(self, a0, x0, a1, x1, a2, x2, a3, x3, out):
        out.value = self.carrier.arithmetic.combine4(a0, x0.value, a1, x1.value, a2, x2.value, a3, x3.value, out.value)
        return out

    def combine5_into(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, out):
        self.carrier.arithmetic.combine5(a0, x0.value, a1, x1.value, a2, x2.value, a3, x3.value, a4, x4.value, out.value)
        return out

    def combine5_return(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, out):
        out.value = self.carrier.arithmetic.combine5(a0, x0.value, a1, x1.value, a2, x2.value, a3, x3.value, a4, x4.value, out.value)
        return out

    def combine6_into(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, out):
        self.carrier.arithmetic.combine6(a0, x0.value, a1, x1.value, a2, x2.value, a3, x3.value, a4, x4.value, a5, x5.value, out.value)
        return out

    def combine6_return(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, out):
        out.value = self.carrier.arithmetic.combine6(a0, x0.value, a1, x1.value, a2, x2.value, a3, x3.value, a4, x4.value, a5, x5.value, out.value)
        return out

    def combine7_into(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, out):
        self.carrier.arithmetic.combine7(a0, x0.value, a1, x1.value, a2, x2.value, a3, x3.value, a4, x4.value, a5, x5.value, a6, x6.value, out.value)
        return out

    def combine7_return(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, out):
        out.value = self.carrier.arithmetic.combine7(a0, x0.value, a1, x1.value, a2, x2.value, a3, x3.value, a4, x4.value, a5, x5.value, a6, x6.value, out.value)
        return out

    def combine8_into(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, out):
        self.carrier.arithmetic.combine8(a0, x0.value, a1, x1.value, a2, x2.value, a3, x3.value, a4, x4.value, a5, x5.value, a6, x6.value, a7, x7.value, out.value)
        return out

    def combine8_return(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, out):
        out.value = self.carrier.arithmetic.combine8(a0, x0.value, a1, x1.value, a2, x2.value, a3, x3.value, a4, x4.value, a5, x5.value, a6, x6.value, a7, x7.value, out.value)
        return out

    def combine9_into(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, out):
        self.carrier.arithmetic.combine9(a0, x0.value, a1, x1.value, a2, x2.value, a3, x3.value, a4, x4.value, a5, x5.value, a6, x6.value, a7, x7.value, a8, x8.value, out.value)
        return out

    def combine9_return(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, out):
        out.value = self.carrier.arithmetic.combine9(a0, x0.value, a1, x1.value, a2, x2.value, a3, x3.value, a4, x4.value, a5, x5.value, a6, x6.value, a7, x7.value, a8, x8.value, out.value)
        return out

    def combine10_into(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, out):
        self.carrier.arithmetic.combine10(a0, x0.value, a1, x1.value, a2, x2.value, a3, x3.value, a4, x4.value, a5, x5.value, a6, x6.value, a7, x7.value, a8, x8.value, a9, x9.value, out.value)
        return out

    def combine10_return(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, out):
        out.value = self.carrier.arithmetic.combine10(a0, x0.value, a1, x1.value, a2, x2.value, a3, x3.value, a4, x4.value, a5, x5.value, a6, x6.value, a7, x7.value, a8, x8.value, a9, x9.value, out.value)
        return out

    def combine11_into(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, a10, x10, out):
        self.carrier.arithmetic.combine11(a0, x0.value, a1, x1.value, a2, x2.value, a3, x3.value, a4, x4.value, a5, x5.value, a6, x6.value, a7, x7.value, a8, x8.value, a9, x9.value, a10, x10.value, out.value)
        return out

    def combine11_return(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, a10, x10, out):
        out.value = self.carrier.arithmetic.combine11(a0, x0.value, a1, x1.value, a2, x2.value, a3, x3.value, a4, x4.value, a5, x5.value, a6, x6.value, a7, x7.value, a8, x8.value, a9, x9.value, a10, x10.value, out.value)
        return out

    def combine12_into(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, a10, x10, a11, x11, out):
        self.carrier.arithmetic.combine12(a0, x0.value, a1, x1.value, a2, x2.value, a3, x3.value, a4, x4.value, a5, x5.value, a6, x6.value, a7, x7.value, a8, x8.value, a9, x9.value, a10, x10.value, a11, x11.value, out.value)
        return out

    def combine12_return(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, a10, x10, a11, x11, out):
        out.value = self.carrier.arithmetic.combine12(a0, x0.value, a1, x1.value, a2, x2.value, a3, x3.value, a4, x4.value, a5, x5.value, a6, x6.value, a7, x7.value, a8, x8.value, a9, x9.value, a10, x10.value, a11, x11.value, out.value)
        return out

    def __add__(self, other: "StarkVectorTranslation") -> "StarkVectorTranslation":
        if self.carrier is not other.carrier:
            raise ValueError("Cannot add translations with different carriers.")

        result = StarkVectorTranslation(self.carrier.allocation.zero_translation(), self.carrier)
        return self.add(other, result)

    def __rmul__(self, scalar: float) -> "StarkVectorTranslation":
        result = StarkVectorTranslation(self.carrier.allocation.zero_translation(), self.carrier)
        return self.scale(scalar, self, result)


class StarkVectorAllocator:
    def __init__(self, carrier: Carrier[Any, Any]) -> None:
        self.carrier = carrier

    def allocate_state(self) -> StarkVector:
        return StarkVector(self.carrier.allocation.zero_state(), self.carrier)

    def copy_state(self, source: StarkVector, out: StarkVector) -> StarkVector:
        out.value = self.carrier.allocation.copy_state(source.value)
        return out

    def allocate_translation(self) -> StarkVectorTranslation:
        return StarkVectorTranslation(self.carrier.allocation.zero_translation(), self.carrier)
