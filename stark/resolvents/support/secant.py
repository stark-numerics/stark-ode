from __future__ import annotations

import numpy as np

from stark.accelerators import AcceleratorAbsent
from stark.contracts import AcceleratorLike, Block
from stark.resolvents.support.workspace import ResolventWorkspace


class SecantLeastSquares:
    """Dense least-squares support for secant-family resolvents."""

    __slots__ = ("depth", "rhs_vector", "gram_matrix")

    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.rhs_vector = np.empty(depth, dtype=np.float64)
        self.gram_matrix = np.empty((depth, depth), dtype=np.float64)

    def solve(
        self,
        count: int,
        inner_product,
        right: list[Block],
        block: Block,
        slot,
    ) -> np.ndarray:
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


class SecantHistory:
    """A fixed-depth rolling history of block-valued secant pairs."""

    __slots__ = (
        "accelerator",
        "workspace",
        "depth",
        "size",
        "count",
        "head",
        "left",
        "right",
        "left_buffer",
        "right_buffer",
        "temporary",
        "project_buffer",
        "least_squares",
    )

    def __init__(self, workspace: ResolventWorkspace, depth: int, accelerator: AcceleratorLike | None = None) -> None:
        if depth < 1:
            raise ValueError("Secant history depth must be at least 1.")
        self.accelerator = accelerator if accelerator is not None else AcceleratorAbsent()
        self.workspace = workspace
        self.depth = depth
        self.size = -1
        self.count = 0
        self.head = 0
        self.left: list[Block] = []
        self.right: list[Block] = []
        self.left_buffer = None
        self.right_buffer = None
        self.temporary = None
        self.project_buffer = None
        self.least_squares = self.accelerator.resolve_support(
            SecantLeastSquares(depth),
            label="resolvent_secant_least_squares",
            depth=depth,
        )

    def bind_accelerator(self, accelerator: AcceleratorLike) -> None:
        self.accelerator = accelerator
        self.least_squares = accelerator.resolve_support(
            SecantLeastSquares(self.depth),
            label="resolvent_secant_least_squares",
            depth=self.depth,
        )

    def __len__(self) -> int:
        return self.count

    def ensure_size(self, size: int) -> None:
        if self.size == size:
            return
        workspace = self.workspace
        self.size = size
        self.count = 0
        self.head = 0
        self.left = [workspace.allocate_block(size) for _ in range(self.depth)]
        self.right = [workspace.allocate_block(size) for _ in range(self.depth)]
        self.left_buffer = workspace.allocate_block(size)
        self.right_buffer = workspace.allocate_block(size)
        self.temporary = workspace.allocate_block(size)
        self.project_buffer = np.empty(self.depth, dtype=np.float64)

    def clear(self) -> None:
        self.count = 0
        self.head = 0

    def append(self, left: Block, right: Block) -> None:
        workspace = self.workspace
        if self.count < self.depth:
            index = (self.head + self.count) % self.depth
            self.count += 1
        else:
            index = self.head
            self.head = (self.head + 1) % self.depth

        workspace.copy_block(self.left[index], left)
        workspace.copy_block(self.right[index], right)

    def append_difference(self, left_now: Block, left_before: Block, right_now: Block, right_before: Block) -> None:
        left_buffer = self.left_buffer
        right_buffer = self.right_buffer
        assert left_buffer is not None
        assert right_buffer is not None
        self.workspace.combine2_block(left_buffer, 1.0, left_now, -1.0, left_before)
        self.workspace.combine2_block(right_buffer, 1.0, right_now, -1.0, right_before)
        self.append(left_buffer, right_buffer)

    def project_right(self, block: Block) -> np.ndarray:
        project_buffer = self.project_buffer
        assert project_buffer is not None
        inner_product = self.workspace.inner_product
        for index in range(self.count):
            project_buffer[index] = inner_product(self.right[self.slot(index)], block)
        return project_buffer[: self.count]

    def solve_right_least_squares(self, block: Block) -> np.ndarray:
        count = self.count
        if count == 0:
            return np.zeros(0, dtype=np.float64)

        return self.least_squares.solve(count, self.workspace.inner_product, self.right, block, self.slot)

    def combine_left(self, out: Block, coefficients: np.ndarray) -> None:
        workspace = self.workspace
        temporary = self.temporary
        assert temporary is not None
        workspace.zero_block(out)
        for index in range(min(self.count, len(coefficients))):
            coefficient = float(coefficients[index])
            if coefficient == 0.0:
                continue
            workspace.combine2_block(temporary, 1.0, out, coefficient, self.left[self.slot(index)])
            workspace.copy_block(out, temporary)

    def slot(self, index: int) -> int:
        return (self.head + index) % self.depth

__all__ = ["SecantHistory", "SecantLeastSquares"]










