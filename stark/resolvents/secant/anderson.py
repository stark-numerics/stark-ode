from __future__ import annotations

"""Anderson-backed resolvent for one-stage shifted implicit solves."""

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from stark.accelerators import AcceleratorAbsent
from stark.block import Block, BlockAllocator
from stark.contracts import AcceleratorLike, InnerProduct, Translation, Allocator
from stark.executor.tolerance import ExecutorTolerance
from stark.resolvents.support import (
    MonitorResolventLike,
    ResolventError,
    ResolventPolicy,
    ResolventSafety,
    ResolventSpecialist,
    ResolventStageProblem,
    ResolventStageResidual,
    ResolventStencilBlock,
    initialise_resolvent_runtime,
    refresh_resolvent_call,
    with_resolvent_call_methods,
    with_resolvent_display,
    with_resolvent_monitoring,
)
from stark.resolvents.support.descriptor import ResolventDescriptor
from stark.resolvents.support.secant import (
    BlockInnerProduct,
    ResolventSecantLeastSquares,
    block_inner_product,
)
from stark.resolvents.support.tolerance import ResolventTolerance


class ResolventAndersonHistory:
    """Anderson-specific history of fixed-point and residual differences.

    The history stores the recent differences used by Anderson acceleration.
    It is algorithmic state, not a block workspace.
    """

    __slots__ = (
        "accelerator",
        "allocator",
        "depth",
        "inner_product",
        "least_squares",
        "size",
        "count",
        "head",
        "fixed_point_differences",
        "residual_differences",
        "previous_fixed_point",
        "previous_residual",
    )

    def __init__(
        self,
        allocator: BlockAllocator[Translation],
        inner_product: BlockInnerProduct,
        depth: int,
        accelerator: AcceleratorLike | None = None,
    ) -> None:
        if type(depth) is not int:
            raise TypeError("Anderson history depth must be an int.")
        if depth < 1:
            raise ValueError("Anderson history depth must be at least 1.")

        self.accelerator = accelerator if accelerator is not None else AcceleratorAbsent()
        self.allocator = allocator
        self.inner_product = inner_product
        self.depth = depth
        self.size = -1
        self.count = 0
        self.head = 0
        self.fixed_point_differences: list[Block[Translation]] = []
        self.residual_differences: list[Block[Translation]] = []
        self.previous_fixed_point: Block[Translation] | None = None
        self.previous_residual: Block[Translation] | None = None
        self.least_squares = self.accelerator.compile(
            ResolventSecantLeastSquares(depth),
            label="resolvent_anderson_least_squares",
        )

    def bind_accelerator(self, accelerator: AcceleratorLike) -> None:
        self.accelerator = accelerator
        self.least_squares = accelerator.compile(
            ResolventSecantLeastSquares(self.depth),
            label="resolvent_anderson_least_squares",
        )

    def __len__(self) -> int:
        return self.count

    def ensure_size(self, block: Block[Translation]) -> None:
        size = len(block)
        if self.size == size:
            return

        self.size = size
        self.count = 0
        self.head = 0
        self.fixed_point_differences = [
            self.allocator.allocate_like(block) for _ in range(self.depth)
        ]
        self.residual_differences = [
            self.allocator.allocate_like(block) for _ in range(self.depth)
        ]
        self.previous_fixed_point = self.allocator.allocate_like(block)
        self.previous_residual = self.allocator.allocate_like(block)

    def clear(self) -> None:
        self.count = 0
        self.head = 0
        self.previous_fixed_point = None
        self.previous_residual = None

    def observe(self, fixed_point: Block[Translation], residual: Block[Translation]) -> None:
        """Record the current fixed-point candidate and residual."""

        self.ensure_size(fixed_point)

        if self.previous_fixed_point is None or self.previous_residual is None:
            self.previous_fixed_point = self.allocator.allocate_like(fixed_point)
            self.previous_residual = self.allocator.allocate_like(residual)
            self.previous_fixed_point.replace(fixed_point)
            self.previous_residual.replace(residual)
            return

        fixed_point_difference = fixed_point - self.previous_fixed_point
        residual_difference = residual - self.previous_residual

        if self.count < self.depth:
            index = (self.head + self.count) % self.depth
            self.count += 1
        else:
            index = self.head
            self.head = (self.head + 1) % self.depth

        self.fixed_point_differences[index].replace(fixed_point_difference)
        self.residual_differences[index].replace(residual_difference)
        self.previous_fixed_point.replace(fixed_point)
        self.previous_residual.replace(residual)

    def correction(
        self,
        residual: Block[Translation],
        out: Block[Translation],
    ) -> bool:
        """Write the Anderson projected correction into ``out``."""

        if self.count == 0:
            return False

        coefficients = self.least_squares.solve(
            self.count,
            self.inner_product,
            self.residual_differences,
            residual,
            self.slot,
        )
        self.combine_fixed_point_differences(out, coefficients)
        return True

    def combine_fixed_point_differences(
        self,
        out: Block[Translation],
        coefficients: np.ndarray,
    ) -> None:
        out.replace(0.0 * out)

        for index in range(min(self.count, len(coefficients))):
            coefficient = float(coefficients[index])
            if coefficient == 0.0:
                continue
            out += coefficient * self.fixed_point_differences[self.slot(index)]

    def slot(self, index: int) -> int:
        return (self.head + index) % self.depth


@with_resolvent_display
@with_resolvent_call_methods
@with_resolvent_monitoring
class ResolventAnderson:
    """Anderson-accelerated fixed-point resolvent.

    Residual equation:

        F(delta) = 0

    Anderson starts from the Picard fixed-point candidate and accelerates it
    using a short history of fixed-point and residual differences.

    Algorithm sketch:

        1. Compute F(delta).
        2. Accept if ||F(delta)|| is within ExecutorTolerance.
        3. Build the Picard fixed-point candidate delta - F(delta).
        4. Record Anderson history from fixed-point/residual differences.
        5. Subtract the projected Anderson correction when history exists.
        6. Recheck once after the final correction.
    """

    __slots__ = (
        "_monitor",
        "accelerator",
        "alpha",
        "allocator",
        "call_pure",
        "fixed_point",
        "history",
        "history_correction",
        "policy",
        "redirect_call",
        "residual",
        "residual_buffer",
        "safety",
        "size",
        "subtract_update",
        "tableau",
        "tolerance",
    )

    descriptor = ResolventDescriptor("Anderson", "Anderson Acceleration")

    if TYPE_CHECKING:
        def assign_monitor(self, monitor: MonitorResolventLike) -> None: ...
        def unassign_monitor(self) -> None: ...
        def record_solve(
            self,
            block_size: int,
            iteration_count: int,
            error: float,
            scale: float,
            converged: bool,
        ) -> None: ...

    def __init__(
        self,
        allocator: Allocator,
        inner_product: InnerProduct,
        ExecutorTolerance: ExecutorTolerance | None = None,
        policy: ResolventPolicy | None = None,
        depth: int = 4,
        safety: ResolventSafety | None = None,
        accelerator: AcceleratorLike | None = None,
        specialist: ResolventSpecialist[Translation] | None = None,
        tableau: Any | None = None,
    ) -> None:
        self.tableau = tableau
        initialise_resolvent_runtime(self, safety, accelerator)

        self.allocator = BlockAllocator(allocator)
        self.tolerance = (
            ExecutorTolerance
            if ExecutorTolerance is not None
            else ResolventTolerance(atol=1.0e-9, rtol=1.0e-9)
        )
        self.policy = policy if policy is not None else ResolventPolicy()
        self.residual = ResolventStageResidual(
            "ResolventAnderson",
            allocator,
            accelerator=self.accelerator,
        )
        self.residual_buffer = None
        self.fixed_point = None
        self.history_correction = None
        self.subtract_update = None
        self.size = -1

        lifted_inner_product = lambda left, right: block_inner_product(
            inner_product,
            left,
            right,
        )
        self.history = ResolventAndersonHistory(
            self.allocator,
            lifted_inner_product,
            depth,
            accelerator=self.accelerator,
        )

        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_pure = self.call_specialized
            refresh_resolvent_call(self)

    def prepare_specialized_kernels(
        self,
        specialist: ResolventSpecialist[Translation],
    ) -> None:
        # Steps 3 and 5 use the same block subtraction stencil.
        self.subtract_update = specialist.provide(
            ResolventStencilBlock((1.0, -1.0))
        )

    def prepare_buffers(self, delta: Block[Translation]) -> None:
        size = len(delta)
        if self.size == size:
            return

        self.size = size
        self.residual_buffer = self.allocator.allocate_like(delta)
        self.fixed_point = self.allocator.allocate_like(delta)
        self.history_correction = self.allocator.allocate_like(delta)
        self.history.ensure_size(delta)

    def call_inline(
        self,
        problem: ResolventStageProblem,
        delta: Block[Translation],
    ) -> Block[Translation]:
        if self.policy.max_iterations < 1:
            raise ValueError("ResolventPolicy.max_iterations must be at least 1.")

        self.alpha = problem.alpha
        self.prepare_buffers(delta)
        self.history.clear()

        F = self.residual
        F.configure(problem)
        residual = cast(Block[Translation], self.residual_buffer)
        fixed_point = cast(Block[Translation], self.fixed_point)
        history_correction = cast(Block[Translation], self.history_correction)

        block_size = len(delta)
        iteration_count = 0

        for _ in range(self.policy.max_iterations):
            # 1. Compute F(delta).
            F(delta, residual)

            # 2. Accept if ||F(delta)|| is within ExecutorTolerance.
            error = residual.norm()
            scale = delta.norm()
            if self.tolerance.accepts(error, scale):
                self.record_solve(block_size, iteration_count, error, scale, True)
                return delta

            # 3. Build the Picard fixed-point candidate delta - F(delta).
            fixed_point.replace(delta - residual)

            # 4. Record Anderson history.
            self.history.observe(fixed_point, residual)

            # 5. Subtract the projected Anderson correction when available.
            if self.history.correction(residual, history_correction):
                delta.replace(fixed_point - history_correction)
            else:
                delta.replace(fixed_point)

            iteration_count += 1

        # 6. Recheck once after the final correction.
        F(delta, residual)

        error = residual.norm()
        scale = delta.norm()
        if self.tolerance.accepts(error, scale):
            self.record_solve(block_size, iteration_count, error, scale, True)
            return delta

        self.record_solve(block_size, iteration_count, error, scale, False)
        raise ResolventError(
            f"{type(self).__name__} failed to resolve the residual within "
            f"{self.policy.max_iterations} iterations (error={error:g})."
        )

    def call_specialized(
        self,
        problem: ResolventStageProblem,
        delta: Block[Translation],
    ) -> Block[Translation]:
        if self.policy.max_iterations < 1:
            raise ValueError("ResolventPolicy.max_iterations must be at least 1.")

        self.alpha = problem.alpha
        self.prepare_buffers(delta)
        self.history.clear()

        F = self.residual
        F.configure(problem)
        residual = cast(Block[Translation], self.residual_buffer)
        fixed_point = cast(Block[Translation], self.fixed_point)
        history_correction = cast(Block[Translation], self.history_correction)
        subtract_update = self.subtract_update
        assert subtract_update is not None

        block_size = len(delta)
        iteration_count = 0

        for _ in range(self.policy.max_iterations):
            # 1. Compute F(delta).
            F(delta, residual)

            # 2. Accept if ||F(delta)|| is within ExecutorTolerance.
            error = residual.norm()
            scale = delta.norm()
            if self.tolerance.accepts(error, scale):
                self.record_solve(block_size, iteration_count, error, scale, True)
                return delta

            # 3. Build the Picard fixed-point candidate delta - F(delta).
            subtract_update(1.0, fixed_point, delta, residual)

            # 4. Record Anderson history.
            self.history.observe(fixed_point, residual)

            # 5. Subtract the projected Anderson correction when available.
            if self.history.correction(residual, history_correction):
                subtract_update(1.0, delta, fixed_point, history_correction)
            else:
                delta.replace(fixed_point)

            iteration_count += 1

        # 6. Recheck once after the final correction.
        F(delta, residual)

        error = residual.norm()
        scale = delta.norm()
        if self.tolerance.accepts(error, scale):
            self.record_solve(block_size, iteration_count, error, scale, True)
            return delta

        self.record_solve(block_size, iteration_count, error, scale, False)
        raise ResolventError(
            f"{type(self).__name__} failed to resolve the residual within "
            f"{self.policy.max_iterations} iterations (error={error:g})."
        )


__all__ = ["ResolventAnderson", "ResolventAndersonHistory"]
