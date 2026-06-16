from __future__ import annotations

from collections.abc import Callable


Vector = list[float]
Matrix = list[float]
InverterNucleusCall = Callable[[Matrix, Vector, Vector], None]


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

    The public call shape is intentionally tiny::

        nucleus(matrix, image, result)

    That makes this class a convenient redirection point for future compiled
    kernels.  A numba/C/Rust kernel only needs the same three-buffer signature;
    the dense inverter does not need to know which implementation is installed.
    """

    __slots__ = (
        "call",
        "dimension",
        "rhs",
        "row_offsets",
        "size",
        "work",
    )

    call: InverterNucleusCall
    dimension: int
    rhs: Vector
    row_offsets: list[int]
    size: int
    work: Matrix

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("InverterNucleus dimension must be positive.")

        self.dimension = dimension
        self.size = dimension * dimension

        # Generic Gaussian elimination needs a private matrix and RHS because
        # it overwrites both.  Allocate them once; repeated tiny allocations are
        # a visible cost in implicit solves.
        self.work = [0.0 for _index in range(self.size)]
        self.rhs = [0.0 for _index in range(dimension)]

        # A row-major flat matrix repeatedly needs ``row * dimension``.  These
        # offsets are small, but they are used in every pivot/elimination loop;
        # precomputing them avoids duplicate multiplication in the generic path.
        self.row_offsets = [row * dimension for row in range(dimension)]

        # Redirect once at construction.  The hot call then pays one attribute
        # lookup and one function call, not a dimension branch.
        if dimension == 1:
            self.call = self.call_1x1
        elif dimension == 2:
            self.call = self.call_2x2
        elif dimension == 3:
            self.call = self.call_3x3
        else:
            self.call = self.call_nxn

    def __call__(self, matrix: Matrix, image: Vector, result: Vector) -> None:
        self.call(matrix, image, result)

    def bind(self, call: InverterNucleusCall) -> None:
        """
        Install an equivalent implementation.

        This is the intended hook for compiled kernels.  The replacement must
        have the same semantics as ``__call__``: read ``matrix`` and ``image``;
        overwrite ``result``; do not resize buffers.
        """

        self.call = call

    def call_1x1(self, matrix: Matrix, image: Vector, result: Vector) -> None:
        # One division is all the 1D case needs.  Let ZeroDivisionError surface;
        # this class is a hot kernel, not a validation boundary.
        result[0] = image[0] / matrix[0]

    def call_2x2(self, matrix: Matrix, image: Vector, result: Vector) -> None:
        # Row-major matrix:
        #   [a b]
        #   [c d]
        a = matrix[0]
        b = matrix[1]
        c = matrix[2]
        d = matrix[3]

        determinant = a * d - b * c
        result[0] = (d * image[0] - b * image[1]) / determinant
        result[1] = (-c * image[0] + a * image[1]) / determinant

    def call_3x3(self, matrix: Matrix, image: Vector, result: Vector) -> None:
        # Row-major matrix:
        #   [a b c]
        #   [d e f]
        #   [g h i]
        #
        # The cofactors below are arranged so that result = inv(matrix) @ image
        # without first materialising the inverse matrix.  This is faster and
        # simpler than generic Python Gaussian elimination for the 3x3 case.
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
        """Solve the generic n-by-n case by Gaussian elimination.

        This routine is written for small dense systems, not for large linear
        algebra.  The goal is to avoid per-call allocations and avoid extra
        abstraction in the inner loops.  For genuinely large systems, STARK
        should use a Krylov or backend-native inverter rather than this nucleus.
        """

        dimension = self.dimension
        row_offsets = self.row_offsets
        work = self.work
        rhs = self.rhs

        # Copy inputs into reusable scratch buffers.  The prepared dense path
        # guarantees correct lengths, so slice assignment is intentionally used
        # here instead of checked element-by-element copying.  It is noticeably
        # faster for the HIRES-size 8x8 case.
        work[:] = matrix
        rhs[:] = image

        # Forward elimination builds an upper triangular matrix in ``work`` and
        # applies the same row operations to ``rhs``.  Partial pivoting protects
        # the simple elimination from obviously poor pivots without introducing
        # a full factorisation object.
        for pivot_index in range(dimension):
            pivot_offset = row_offsets[pivot_index]
            pivot_row = pivot_index
            pivot_abs = abs(work[pivot_offset + pivot_index])

            # Find the largest available pivot in this column.
            for row in range(pivot_index + 1, dimension):
                candidate_abs = abs(work[row_offsets[row] + pivot_index])
                if candidate_abs > pivot_abs:
                    pivot_abs = candidate_abs
                    pivot_row = row

            if pivot_abs == 0.0:
                raise ZeroDivisionError("Inverter matrix is singular.")

            # Swap only the active row suffix.  Columns before pivot_index have
            # already served their purpose and are not read by later pivots or
            # back substitution, so swapping them is wasted work.
            if pivot_row != pivot_index:
                swap_offset = row_offsets[pivot_row]
                for column in range(pivot_index, dimension):
                    pivot_entry = pivot_offset + column
                    swap_entry = swap_offset + column
                    work[pivot_entry], work[swap_entry] = (
                        work[swap_entry],
                        work[pivot_entry],
                    )
                rhs[pivot_index], rhs[pivot_row] = rhs[pivot_row], rhs[pivot_index]

            pivot = work[pivot_offset + pivot_index]

            for row in range(pivot_index + 1, dimension):
                row_offset = row_offsets[row]
                factor = work[row_offset + pivot_index] / pivot

                # Keep the triangular structure explicit.  This entry is not
                # needed by back substitution, but setting it to zero makes the
                # scratch matrix easier to inspect when debugging numerical
                # failures.
                work[row_offset + pivot_index] = 0.0

                for column in range(pivot_index + 1, dimension):
                    work[row_offset + column] -= factor * work[pivot_offset + column]

                rhs[row] -= factor * rhs[pivot_index]

        # Back substitution writes into ``result`` from high row to low row.
        # Values at columns greater than the current row have already been
        # produced by this call, so ``result`` does not need clearing first.
        for row in range(dimension - 1, -1, -1):
            row_offset = row_offsets[row]
            total = rhs[row]

            for column in range(row + 1, dimension):
                total -= work[row_offset + column] * result[column]

            pivot = work[row_offset + row]
            if pivot == 0.0:
                raise ZeroDivisionError("Inverter matrix is singular.")

            result[row] = total / pivot


__all__ = ["InverterNucleus", "InverterNucleusCall"]
