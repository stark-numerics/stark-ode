from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass, field

from stark.accelerators import DEFAULT_ACCELERATOR
from stark.contracts.accelerator import AcceleratorLike


def _invert_general(
    matrix: Sequence[float],
    image: Sequence[float],
    result: MutableSequence[float],
) -> MutableSequence[float]:
    """Dependency-free Gaussian elimination with partial pivoting."""

    dimension = len(image)
    work = [float(matrix[index]) for index in range(dimension * dimension)]
    rhs = [float(image[row]) for row in range(dimension)]

    for pivot_index in range(dimension):
        pivot_row = pivot_index
        pivot_abs = abs(work[pivot_row * dimension + pivot_index])
        for row in range(pivot_index + 1, dimension):
            candidate_abs = abs(work[row * dimension + pivot_index])
            if candidate_abs > pivot_abs:
                pivot_abs = candidate_abs
                pivot_row = row

        if pivot_abs == 0.0:
            raise ZeroDivisionError("Dense matrix is singular.")

        if pivot_row != pivot_index:
            for column in range(dimension):
                pivot_offset = pivot_index * dimension + column
                swap_offset = pivot_row * dimension + column
                work[pivot_offset], work[swap_offset] = work[swap_offset], work[pivot_offset]
            rhs[pivot_index], rhs[pivot_row] = rhs[pivot_row], rhs[pivot_index]

        pivot = work[pivot_index * dimension + pivot_index]
        for row in range(pivot_index + 1, dimension):
            factor = work[row * dimension + pivot_index] / pivot
            work[row * dimension + pivot_index] = 0.0
            for column in range(pivot_index + 1, dimension):
                work[row * dimension + column] -= factor * work[pivot_index * dimension + column]
            rhs[row] -= factor * rhs[pivot_index]

    for row in range(dimension - 1, -1, -1):
        total = rhs[row]
        for column in range(row + 1, dimension):
            total -= work[row * dimension + column] * result[column]

        pivot = work[row * dimension + row]
        if pivot == 0.0:
            raise ZeroDivisionError("Dense matrix is singular.")
        result[row] = total / pivot

    return result


@dataclass(slots=True)
class InverterProviderDenseNativeGeneral:
    """Private dependency-free fallback for dense systems larger than three."""

    dimension: int
    accelerator: AcceleratorLike = field(default_factory=lambda: DEFAULT_ACCELERATOR)
    invert: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dimension <= 3:
            raise ValueError("Native general dense fallback is only used for dimension greater than three.")

        self.invert = self.accelerator.compile(
            _invert_general,
            label="inverter-provider-dense-native-general",
        )

    def __call__(
        self,
        matrix: Sequence[float],
        image: Sequence[float],
        result: MutableSequence[float],
    ) -> MutableSequence[float]:
        return self.invert(matrix, image, result)  # type: ignore[misc]


__all__ = ["InverterProviderDenseNativeGeneral"]
