from __future__ import annotations

"""
Reusable secant-history support for nonlinear resolvers.

Many nonlinear solvers need to retain differences of iterates or residuals:

- Anderson acceleration stores residual and fixed-point differences
- Broyden-type methods store secant updates for approximate Jacobian or inverse
  Jacobian actions

This module keeps that block-level bookkeeping in one place so resolver code can
focus on the actual method.
"""

import numpy as np

from stark.contracts import Block
from stark.resolver_support.workspace import ResolverWorkspace


class SecantHistory:
    """A fixed-depth rolling history of block-valued secant pairs."""

    __slots__ = (
        "workspace",
        "depth",
        "size",
        "count",
        "left",
        "right",
        "left_buffer",
        "right_buffer",
        "temporary",
    )

    def __init__(self, workspace: ResolverWorkspace, depth: int) -> None:
        if depth < 1:
            raise ValueError("Secant history depth must be at least 1.")
        self.workspace = workspace
        self.depth = depth
        self.size = -1
        self.count = 0
        self.left: list[Block] = []
        self.right: list[Block] = []
        self.left_buffer = None
        self.right_buffer = None
        self.temporary = None

    def __len__(self) -> int:
        return self.count

    def ensure_size(self, size: int) -> None:
        """Allocate or resize all stored secant blocks for a given block length."""
        if self.size == size:
            return
        workspace = self.workspace
        self.size = size
        self.count = 0
        self.left = [workspace.allocate_block(size) for _ in range(self.depth)]
        self.right = [workspace.allocate_block(size) for _ in range(self.depth)]
        self.left_buffer = workspace.allocate_block(size)
        self.right_buffer = workspace.allocate_block(size)
        self.temporary = workspace.allocate_block(size)

    def clear(self) -> None:
        """Forget all stored secant pairs while keeping the allocated buffers."""
        self.count = 0

    def append(self, left: Block, right: Block) -> None:
        """Append one secant pair, discarding the oldest pair if the history is full."""
        workspace = self.workspace
        if self.count < self.depth:
            index = self.count
            self.count += 1
        else:
            for index in range(self.depth - 1):
                workspace.copy_block(self.left[index], self.left[index + 1])
                workspace.copy_block(self.right[index], self.right[index + 1])
            index = self.depth - 1

        workspace.copy_block(self.left[index], left)
        workspace.copy_block(self.right[index], right)

    def append_difference(self, left_now: Block, left_before: Block, right_now: Block, right_before: Block) -> None:
        """Append the secant pair built from differences of two block pairs."""
        left_buffer = self.left_buffer
        right_buffer = self.right_buffer
        assert left_buffer is not None
        assert right_buffer is not None
        self.workspace.combine2_block(left_buffer, 1.0, left_now, -1.0, left_before)
        self.workspace.combine2_block(right_buffer, 1.0, right_now, -1.0, right_before)
        self.append(left_buffer, right_buffer)

    def project_right(self, block: Block) -> np.ndarray:
        """Return the inner products of the stored right blocks against `block`."""
        return np.array(
            [self.workspace.inner_product(self.right[index], block) for index in range(self.count)],
            dtype=np.float64,
        )

    def solve_right_least_squares(self, block: Block) -> np.ndarray:
        """
        Solve the normal equations for the stored right blocks against `block`.

        This returns coefficients `gamma` minimizing the norm of

            block - sum_i gamma_i right_i
        """
        count = self.count
        if count == 0:
            return np.zeros(0, dtype=np.float64)

        gram = np.empty((count, count), dtype=np.float64)
        rhs = np.empty(count, dtype=np.float64)
        for row in range(count):
            rhs[row] = self.workspace.inner_product(self.right[row], block)
            for column in range(count):
                gram[row, column] = self.workspace.inner_product(self.right[row], self.right[column])

        try:
            return np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(gram, rhs, rcond=None)[0]

    def combine_left(self, out: Block, coefficients: np.ndarray) -> None:
        """Fill `out` with the linear combination of stored left blocks."""
        workspace = self.workspace
        temporary = self.temporary
        assert temporary is not None
        workspace.zero_block(out)
        for index in range(min(self.count, len(coefficients))):
            coefficient = float(coefficients[index])
            if coefficient == 0.0:
                continue
            workspace.combine2_block(temporary, 1.0, out, coefficient, self.left[index])
            workspace.copy_block(out, temporary)


__all__ = ["SecantHistory"]
