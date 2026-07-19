from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from stark.core.contracts import Accelerator


Vector = list[float]
Matrix = list[float]
InverterNucleusCall = Callable[[Matrix, Vector, Vector], None]
InverterNucleusKernel = Callable[[int, list[int], Matrix, Vector, Matrix, Vector, Vector], None]
InverterNucleusFactorCall = Callable[[Vector, Vector], None]


class InverterNucleusAccelerator(Protocol):
    """Small part of the Accelerator protocol used by the nucleus."""

    name: str
    strict: bool

    def compile(
        self,
        function: InverterNucleusKernel,
        /,
        *,
        label: str | None = None,
        cache: bool | None = None,
        **options: Any,
    ) -> InverterNucleusKernel: ...


def _call_nxn_kernel(
    dimension: int,
    row_offsets: list[int],
    work: Matrix,
    rhs: Vector,
    matrix: Matrix,
    image: Vector,
    result: Vector,
) -> None:
    """Imperative Gaussian-elimination kernel used by the generic nucleus.

    This is a one-shot solve: it copies the supplied matrix into ``work``,
    factors it, applies the same row operations to ``rhs``, then back
    substitutes.  Block-operator-bound dense inverter instances use
    ``InverterNucleusFactor`` instead, so chord/very-chord paths factor once
    and solve many right-hand sides.
    """

    size = dimension * dimension
    for index in range(size):
        work[index] = matrix[index]
    for row in range(dimension):
        rhs[row] = image[row]

    for pivot_index in range(dimension):
        pivot_offset = row_offsets[pivot_index]
        pivot_row = pivot_index
        pivot_abs = abs(work[pivot_offset + pivot_index])

        for row in range(pivot_index + 1, dimension):
            candidate_abs = abs(work[row_offsets[row] + pivot_index])
            if candidate_abs > pivot_abs:
                pivot_abs = candidate_abs
                pivot_row = row

        if pivot_abs == 0.0:
            raise ZeroDivisionError("Inverter matrix is singular.")

        if pivot_row != pivot_index:
            swap_offset = row_offsets[pivot_row]
            for column in range(pivot_index, dimension):
                pivot_entry = pivot_offset + column
                swap_entry = swap_offset + column
                work[pivot_entry], work[swap_entry] = work[swap_entry], work[pivot_entry]
            rhs[pivot_index], rhs[pivot_row] = rhs[pivot_row], rhs[pivot_index]

        pivot = work[pivot_offset + pivot_index]

        for row in range(pivot_index + 1, dimension):
            row_offset = row_offsets[row]
            factor = work[row_offset + pivot_index] / pivot
            work[row_offset + pivot_index] = 0.0

            for column in range(pivot_index + 1, dimension):
                work[row_offset + column] -= factor * work[pivot_offset + column]

            rhs[row] -= factor * rhs[pivot_index]

    for row in range(dimension - 1, -1, -1):
        row_offset = row_offsets[row]
        total = rhs[row]

        for column in range(row + 1, dimension):
            total -= work[row_offset + column] * result[column]

        pivot = work[row_offset + row]
        if pivot == 0.0:
            raise ZeroDivisionError("Inverter matrix is singular.")

        result[row] = total / pivot


class InverterNucleusFactor:
    """Prepared inverse action for one fixed dense matrix.

    ``InverterNucleus`` is the one-shot solve worker.  This factor object is the
    cached form used by operator-bound inverter instances: the matrix is copied
    and factored once at instance construction, then each correction solve only
    applies the stored factorisation to a new image vector.

    The implementation deliberately remains pure Python/list based.  That keeps
    the dense inverter free of a hidden NumPy dependency while still exposing a
    clean place for a later accelerator-backed factor implementation.
    """

    __slots__ = (
        "dimension",
        "inverse",
        "lu",
        "pivot_rows",
        "redirect_call",
        "rhs",
        "row_offsets",
        "size",
    )

    dimension: int
    inverse: Matrix
    lu: Matrix
    pivot_rows: list[int]
    redirect_call: InverterNucleusFactorCall
    rhs: Vector
    row_offsets: list[int]
    size: int

    def __init__(self, dimension: int, row_offsets: list[int], matrix: Matrix) -> None:
        self.dimension = dimension
        self.size = dimension * dimension
        self.row_offsets = row_offsets
        self.rhs = [0.0 for _index in range(dimension)]
        self.lu = [0.0 for _index in range(self.size)]
        self.inverse = []
        self.pivot_rows = []

        if dimension == 1:
            self.prepare_1x1(matrix)
            self.redirect_call = self.call_1x1
        elif dimension == 2:
            self.prepare_2x2(matrix)
            self.redirect_call = self.call_inverse
        elif dimension == 3:
            self.prepare_3x3(matrix)
            self.redirect_call = self.call_inverse
        else:
            self.prepare_nxn(matrix)
            self.redirect_call = self.call_nxn

    def __call__(self, image: Vector, result: Vector) -> None:
        self.redirect_call(image, result)

    def prepare_1x1(self, matrix: Matrix) -> None:
        self.inverse = [1.0 / matrix[0]]

    def call_1x1(self, image: Vector, result: Vector) -> None:
        result[0] = image[0] * self.inverse[0]

    def prepare_2x2(self, matrix: Matrix) -> None:
        a = matrix[0]
        b = matrix[1]
        c = matrix[2]
        d = matrix[3]
        determinant = a * d - b * c
        inverse_scale = 1.0 / determinant
        self.inverse = [
            d * inverse_scale,
            -b * inverse_scale,
            -c * inverse_scale,
            a * inverse_scale,
        ]

    def prepare_3x3(self, matrix: Matrix) -> None:
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
        inverse_scale = 1.0 / determinant
        # These entries match the row-major inverse matrix used by
        # ``call_inverse``.  The direct 3x3 one-shot kernel computes the same
        # dot products but recomputes the cofactors every solve; the factor
        # caches them for chord/very-chord reuse.
        self.inverse = [
            cofactor00 * inverse_scale,
            cofactor01 * inverse_scale,
            cofactor02 * inverse_scale,
            cofactor10 * inverse_scale,
            cofactor11 * inverse_scale,
            cofactor12 * inverse_scale,
            cofactor20 * inverse_scale,
            cofactor21 * inverse_scale,
            cofactor22 * inverse_scale,
        ]

    def call_inverse(self, image: Vector, result: Vector) -> None:
        dimension = self.dimension
        inverse = self.inverse
        for row in range(dimension):
            offset = row * dimension
            total = 0.0
            for column in range(dimension):
                total += inverse[offset + column] * image[column]
            result[row] = total

    def prepare_nxn(self, matrix: Matrix) -> None:
        """Factor the matrix into a compact LU form with partial pivoting.

        The lower-triangular multipliers are stored below the diagonal and the
        upper-triangular matrix is stored on/above it.  ``pivot_rows`` records
        the row swaps performed during factorisation so a new right-hand side
        can be permuted in the same way before forward/back substitution.
        """

        lu = self.lu
        for index in range(self.size):
            lu[index] = matrix[index]

        pivot_rows = []
        dimension = self.dimension
        row_offsets = self.row_offsets

        for pivot_index in range(dimension):
            pivot_offset = row_offsets[pivot_index]
            pivot_row = pivot_index
            pivot_abs = abs(lu[pivot_offset + pivot_index])

            for row in range(pivot_index + 1, dimension):
                candidate_abs = abs(lu[row_offsets[row] + pivot_index])
                if candidate_abs > pivot_abs:
                    pivot_abs = candidate_abs
                    pivot_row = row

            if pivot_abs == 0.0:
                raise ZeroDivisionError("Inverter matrix is singular.")

            pivot_rows.append(pivot_row)

            if pivot_row != pivot_index:
                swap_offset = row_offsets[pivot_row]
                # For a reusable LU factorisation the entire active row matters:
                # columns before ``pivot_index`` contain previous L multipliers,
                # and columns from ``pivot_index`` onward contain the active U
                # row.  Swap the whole row so later solves see the same P*L*U.
                for column in range(dimension):
                    pivot_entry = pivot_offset + column
                    swap_entry = swap_offset + column
                    lu[pivot_entry], lu[swap_entry] = lu[swap_entry], lu[pivot_entry]

            pivot = lu[pivot_offset + pivot_index]

            for row in range(pivot_index + 1, dimension):
                row_offset = row_offsets[row]
                factor = lu[row_offset + pivot_index] / pivot
                lu[row_offset + pivot_index] = factor

                for column in range(pivot_index + 1, dimension):
                    lu[row_offset + column] -= factor * lu[pivot_offset + column]

        self.pivot_rows = pivot_rows

    def call_nxn(self, image: Vector, result: Vector) -> None:
        dimension = self.dimension
        row_offsets = self.row_offsets
        lu = self.lu
        rhs = self.rhs

        for row in range(dimension):
            rhs[row] = image[row]

        # Apply the same row interchanges that were chosen during LU
        # factorisation: P * image.
        for pivot_index, pivot_row in enumerate(self.pivot_rows):
            if pivot_row != pivot_index:
                rhs[pivot_index], rhs[pivot_row] = rhs[pivot_row], rhs[pivot_index]

        # Forward substitution through unit-lower L.
        for row in range(dimension):
            row_offset = row_offsets[row]
            total = rhs[row]
            for column in range(row):
                total -= lu[row_offset + column] * rhs[column]
            rhs[row] = total

        # Back substitution through upper U.
        for row in range(dimension - 1, -1, -1):
            row_offset = row_offsets[row]
            total = rhs[row]
            for column in range(row + 1, dimension):
                total -= lu[row_offset + column] * result[column]
            pivot = lu[row_offset + row]
            if pivot == 0.0:
                raise ZeroDivisionError("Inverter matrix is singular.")
            result[row] = total / pivot


class InverterNucleus:
    """
    Small dense inverse-action kernel.

    The nucleus solves the row-major coordinate system::

        sum(matrix[row * dimension + column] * result[column]) == image[row]

    It is deliberately stateful.  The surrounding dense inverter reuses a
    nucleus for many solves with the same dimension, so setup work belongs in
    ``__init__`` rather than in the hot call.

    Dimensions 1, 2, and 3 use direct formulae.  Larger dimensions use Gaussian
    elimination with partial pivoting and reusable scratch buffers.

    Acceleration remains optional and narrow.  The default path is pure Python
    and uses only list-like buffers, so adding an accelerator never introduces a
    hidden NumPy dependency.  The current accelerated path is still a best-effort
    imperative list-buffer kernel for n-by-n one-shot solves; factor reuse is the
    more important optimisation for chord/very-chord dense instances.
    """

    __slots__ = (
        "dimension",
        "kernel",
        "redirect_call",
        "rhs",
        "row_offsets",
        "size",
        "work",
    )

    dimension: int
    kernel: InverterNucleusKernel | None
    redirect_call: InverterNucleusCall
    rhs: Vector
    row_offsets: list[int]
    size: int
    work: Matrix

    def __init__(self, dimension: int, accelerator: Accelerator | None = None):
        if dimension <= 0:
            raise ValueError("InverterNucleus dimension must be positive.")

        self.dimension = dimension
        self.size = dimension * dimension
        self.work = [0.0 for _index in range(self.size)]
        self.rhs = [0.0 for _index in range(dimension)]
        self.row_offsets = [row * dimension for row in range(dimension)]
        self.kernel = None

        if dimension == 1:
            self.redirect_call = self.call_1x1
        elif dimension == 2:
            self.redirect_call = self.call_2x2
        elif dimension == 3:
            self.redirect_call = self.call_3x3
        else:
            self.redirect_call = self.call_nxn
            self.prepare_accelerated_kernel(accelerator)

    def __call__(self, matrix: Matrix, image: Vector, result: Vector) -> None:
        self.redirect_call(matrix, image, result)

    def factor(self, matrix: Matrix) -> InverterNucleusFactor:
        """Prepare a reusable inverse action for a fixed matrix.

        Dense ``instance(operator)`` paths call this once when a chord-like
        resolvent freezes an operator.  Later correction solves reuse the cached
        inverse/cofactor/LU data instead of refactoring the matrix for every
        right-hand side.
        """

        return InverterNucleusFactor(self.dimension, self.row_offsets, matrix)

    def prepare_accelerated_kernel(self, accelerator: Accelerator | None) -> None:
        """Install an accelerated generic kernel when the accelerator is suitable."""

        if accelerator is None:
            return

        name = getattr(accelerator, "name", "")
        if name in ("", "none"):
            return

        if name != "numba":
            if getattr(accelerator, "strict", False):
                raise RuntimeError(
                    "InverterNucleus acceleration currently supports imperative "
                    "list-buffer accelerators such as numba."
                )
            return

        try:
            compiled = accelerator.compile(
                _call_nxn_kernel,
                label=f"inverter-nucleus-{self.dimension}x{self.dimension}",
                cache=True,
            )
        except Exception:
            if getattr(accelerator, "strict", False):
                raise
            return

        self.kernel = compiled
        self.redirect_call = self.call_nxn_accelerated

    def call_1x1(self, matrix: Matrix, image: Vector, result: Vector) -> None:
        result[0] = image[0] / matrix[0]

    def call_2x2(self, matrix: Matrix, image: Vector, result: Vector) -> None:
        a = matrix[0]
        b = matrix[1]
        c = matrix[2]
        d = matrix[3]

        determinant = a * d - b * c
        result[0] = (d * image[0] - b * image[1]) / determinant
        result[1] = (-c * image[0] + a * image[1]) / determinant

    def call_3x3(self, matrix: Matrix, image: Vector, result: Vector) -> None:
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

        result[0] = (
            cofactor00 * image[0]
            + cofactor01 * image[1]
            + cofactor02 * image[2]
        ) / determinant
        result[1] = (
            cofactor10 * image[0]
            + cofactor11 * image[1]
            + cofactor12 * image[2]
        ) / determinant
        result[2] = (
            cofactor20 * image[0]
            + cofactor21 * image[1]
            + cofactor22 * image[2]
        ) / determinant

    def call_nxn(self, matrix: Matrix, image: Vector, result: Vector) -> None:
        _call_nxn_kernel(
            self.dimension,
            self.row_offsets,
            self.work,
            self.rhs,
            matrix,
            image,
            result,
        )

    def call_nxn_accelerated(self, matrix: Matrix, image: Vector, result: Vector) -> None:
        kernel = self.kernel
        assert kernel is not None
        kernel(
            self.dimension,
            self.row_offsets,
            self.work,
            self.rhs,
            matrix,
            image,
            result,
        )


__all__ = [
    "InverterNucleus",
    "InverterNucleusCall",
    "InverterNucleusFactor",
    "InverterNucleusFactorCall",
    "InverterNucleusKernel",
]
