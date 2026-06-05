from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass, field
from typing import Callable

from stark.accelerators import AcceleratorNone
from stark.contracts.accelerator import Accelerator
from stark.inverters.dense.native_general import InverterProviderDenseNativeGeneral

DenseInvert = Callable[
    [Sequence[float], Sequence[float], MutableSequence[float]],
    MutableSequence[float],
]


def _invert_scalar(
    matrix: Sequence[float],
    image: Sequence[float],
    result: MutableSequence[float],
) -> MutableSequence[float]:
    pivot = matrix[0]
    if pivot == 0.0:
        raise ZeroDivisionError("Dense scalar matrix is singular.")
    result[0] = image[0] / pivot
    return result


def _invert_two(
    matrix: Sequence[float],
    image: Sequence[float],
    result: MutableSequence[float],
) -> MutableSequence[float]:
    a = matrix[0]
    b = matrix[1]
    c = matrix[2]
    d = matrix[3]
    determinant = a * d - b * c
    if determinant == 0.0:
        raise ZeroDivisionError("Dense two-by-two matrix is singular.")

    result[0] = (d * image[0] - b * image[1]) / determinant
    result[1] = (-c * image[0] + a * image[1]) / determinant
    return result


def _invert_three(
    matrix: Sequence[float],
    image: Sequence[float],
    result: MutableSequence[float],
) -> MutableSequence[float]:
    a = matrix[0]
    b = matrix[1]
    c = matrix[2]
    d = matrix[3]
    e = matrix[4]
    f = matrix[5]
    g = matrix[6]
    h = matrix[7]
    i = matrix[8]

    cofactor00 = e * i - f * h
    cofactor01 = c * h - b * i
    cofactor02 = b * f - c * e
    cofactor10 = f * g - d * i
    cofactor11 = a * i - c * g
    cofactor12 = c * d - a * f
    cofactor20 = d * h - e * g
    cofactor21 = b * g - a * h
    cofactor22 = a * e - b * d

    determinant = a * cofactor00 + b * cofactor10 + c * cofactor20
    if determinant == 0.0:
        raise ZeroDivisionError("Dense three-by-three matrix is singular.")

    result[0] = (cofactor00 * image[0] + cofactor01 * image[1] + cofactor02 * image[2]) / determinant
    result[1] = (cofactor10 * image[0] + cofactor11 * image[1] + cofactor12 * image[2]) / determinant
    result[2] = (cofactor20 * image[0] + cofactor21 * image[1] + cofactor22 * image[2]) / determinant
    return result


@dataclass(slots=True)
class InverterProviderDenseNative:
    """
    Dependency-free dense inversion provider.

    The provider prepares once for a dimension, redirects scalar, two-by-two,
    and three-by-three systems to exact closed-form kernels, and delegates
    larger systems to a private Gaussian-elimination fallback. If an accelerator
    is supplied, the selected native kernel is compiled during preparation.
    """

    accelerator: Accelerator = field(default_factory=AcceleratorNone)
    dimension: int | None = field(init=False, default=None)
    invert: DenseInvert = field(init=False, repr=False)
    general: InverterProviderDenseNativeGeneral | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.invert = self.invert_unprepared

    def prepare(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("Dense provider dimension must be positive.")
        if self.dimension == dimension:
            return
        if self.dimension is not None:
            raise ValueError("Dense provider has already been prepared for a different dimension.")

        self.dimension = dimension
        if dimension == 1:
            self.invert = self.accelerator.compile(_invert_scalar, label="inverter-provider-dense-native-scalar")
        elif dimension == 2:
            self.invert = self.accelerator.compile(_invert_two, label="inverter-provider-dense-native-two")
        elif dimension == 3:
            self.invert = self.accelerator.compile(_invert_three, label="inverter-provider-dense-native-three")
        else:
            self.general = InverterProviderDenseNativeGeneral(dimension, accelerator=self.accelerator)
            self.invert = self.general

    def invert_unprepared(
        self,
        matrix: Sequence[float],
        image: Sequence[float],
        result: MutableSequence[float],
    ) -> MutableSequence[float]:
        del matrix, image, result
        raise RuntimeError("Dense provider must be prepared before use.")


__all__ = ["InverterProviderDenseNative"]
