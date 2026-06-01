from __future__ import annotations

"""Small dense helpers for resolvent secant-family methods.

This module intentionally avoids owning block allocation or block algebra.
Anderson and Broyden keep their algorithm-specific histories alongside the
resolvent classes that use them.
"""

from collections.abc import Callable
from typing import TypeAlias

import numpy as np

from stark.block import Block
from stark.contracts import Translation


BlockInnerProduct: TypeAlias = Callable[[Block[Translation], Block[Translation]], float]


def block_inner_product(entry_inner_product, left: Block[Translation], right: Block[Translation]) -> float:
    """Lift an entry inner product to blocks."""

    Block._require_same_size(left, right)
    total = 0.0
    for left_item, right_item in zip(left, right, strict=True):
        total += float(entry_inner_product(left_item, right_item))
    return total


class ResolventSecantLeastSquares:
    """Dense least-squares support for small secant histories."""

    __slots__ = ("depth", "rhs_vector", "gram_matrix")

    def __init__(self, depth: int) -> None:
        if type(depth) is not int:
            raise TypeError("Secant least-squares depth must be an int.")
        if depth < 1:
            raise ValueError("Secant least-squares depth must be at least 1.")

        self.depth = depth
        self.rhs_vector = np.empty(depth, dtype=np.float64)
        self.gram_matrix = np.empty((depth, depth), dtype=np.float64)

    def solve(
        self,
        count: int,
        inner_product: BlockInnerProduct,
        right: list[Block[Translation]],
        block: Block[Translation],
        slot,
    ) -> np.ndarray:
        if count == 0:
            return np.zeros(0, dtype=np.float64)

        rhs = self.rhs_vector
        gram = self.gram_matrix

        for row in range(count):
            right_row = right[slot(row)]
            rhs[row] = inner_product(right_row, block)

            for column in range(count):
                gram[row, column] = inner_product(right_row, right[slot(column)])

        rhs_view = rhs[:count]
        gram_view = gram[:count, :count]

        try:
            return np.linalg.solve(gram_view, rhs_view)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(gram_view, rhs_view, rcond=None)[0]


__all__ = [
    "BlockInnerProduct",
    "ResolventSecantLeastSquares",
    "block_inner_product",
]
