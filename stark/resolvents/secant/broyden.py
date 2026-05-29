from __future__ import annotations

"""Broyden-backed resolvent for one-stage shifted implicit solves."""

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from stark.block import Block, BlockAllocator
from stark.contracts import AcceleratorLike, InnerProduct, Translation, Allocator
from stark.accelerators import AcceleratorAbsent
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
    with_resolvent_display,
    with_resolvent_monitoring,
)
from stark.resolvents.support.descriptor import ResolventDescriptor
from stark.resolvents.support.secant import BlockInnerProduct, block_inner_product
from stark.resolvents.support.tolerance import ResolventTolerance
from stark.resolvents.support.safety import ResolventSafety, ResolventSafetyDefault


class ResolventBroydenHistory:
    """Inverse-Broyden history of low-rank inverse updates."""

    __slots__ = (
        "allocator",
        "depth",
        "inner_product",
        "size",
        "count",
        "head",
        "left",
        "right",
    )

    def __init__(
        self,
        allocator: BlockAllocator[Translation],
        inner_product: BlockInnerProduct,
        depth: int,
    ) -> None:
        if type(depth) is not int:
            raise TypeError("Broyden history depth must be an int.")
        if depth < 1:
            raise ValueError("Broyden history depth must be at least 1.")

        self.allocator = allocator
        self.inner_product = inner_product
        self.depth = depth
        self.size = -1
        self.count = 0
        self.head = 0
        self.left: list[Block[Translation]] = []
        self.right: list[Block[Translation]] = []

    def __len__(self) -> int:
        return self.count

    def ensure_size(self, block: Block[Translation]) -> None:
        size = len(block)
        if self.size == size:
            return

        self.size = size
        self.count = 0
        self.head = 0
        self.left = [self.allocator.allocate_like(block) for _ in range(self.depth)]
        self.right = [self.allocator.allocate_like(block) for _ in range(self.depth)]

    def clear(self) -> None:
        self.count = 0
        self.head = 0

    def append(self, left: Block[Translation], right: Block[Translation]) -> None:
        self.ensure_size(left)

        if self.count < self.depth:
            index = (self.head + self.count) % self.depth
            self.count += 1
        else:
            index = self.head
            self.head = (self.head + 1) % self.depth

        self.left[index].replace(left)
        self.right[index].replace(right)

    def project_right(self, block: Block[Translation]) -> np.ndarray:
        coefficients = np.empty(self.count, dtype=np.float64)
        for index in range(self.count):
            coefficients[index] = self.inner_product(self.right[self.slot(index)], block)
        return coefficients

    def combine_left(
        self,
        out: Block[Translation],
        coefficients: np.ndarray,
    ) -> None:
        out.replace(0.0 * out)

        for index in range(min(self.count, len(coefficients))):
            coefficient = float(coefficients[index])
            if coefficient == 0.0:
                continue
            out += coefficient * self.left[self.slot(index)]

    def apply_inverse(
        self,
        block: Block[Translation],
        history_correction: Block[Translation],
        out: Block[Translation],
    ) -> None:
        out.replace(block)
        if self.count == 0:
            return

        coefficients = self.project_right(block)
        self.combine_left(history_correction, coefficients)
        out += history_correction

    def slot(self, index: int) -> int:
        return (self.head + index) % self.depth


@with_resolvent_display
@with_resolvent_monitoring
class ResolventBroyden:
    """Inverse-Broyden resolvent for one-stage shifted implicit equations.

    Residual equation:

        F(delta) = 0

    Broyden maintains a low-rank approximation to the inverse residual
    differential and uses it to propose nonlinear corrections.

    Algorithm sketch:

        1. Compute F(delta).
        2. Accept if ||F(delta)|| is within ExecutorTolerance.
        3. Apply the inverse approximation to propose a correction.
        4. Trial the correction and compute the residual change.
        5. Store a new inverse-Broyden secant pair when informative.
        6. Accept the trial iterate.
        7. Recheck once after the final correction.
    """

    __slots__ = (
        "_monitor",
        "accelerator",
        "add_update",
        "alpha",
        "allocator",
        "call_monitorable",
        "correction",
        "difference_update",
        "history",
        "history_correction",
        "inner_product",
        "inverse_residual_delta",
        "next_residual",
        "policy",
        "redirect_call",
        "residual",
        "residual_buffer",
        "residual_delta",
        "safety",
        "scaled_update",
        "size",
        "tableau",
        "tolerance",
        "trial",
    )

    descriptor = ResolventDescriptor("Broyden", "Inverse Broyden")

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
        depth: int = 8,
        safety: ResolventSafety | None = None,
        accelerator: AcceleratorLike | None = None,
        specialist: ResolventSpecialist[Translation] | None = None,
        tableau: Any | None = None,
    ) -> None:
        self.tableau = tableau
        self.safety = safety if safety is not None else ResolventSafetyDefault()
        self.accelerator = accelerator if accelerator is not None else AcceleratorAbsent()
        self.alpha = 0.0
        self._monitor = None

        self.allocator = BlockAllocator(allocator)
        self.tolerance = (
            ExecutorTolerance
            if ExecutorTolerance is not None
            else ResolventTolerance(atol=1.0e-9, rtol=1.0e-9)
        )
        self.policy = policy if policy is not None else ResolventPolicy()
        self.residual = ResolventStageResidual(
            "ResolventBroyden",
            allocator,
            accelerator=self.accelerator,
        )
        self.residual_buffer = None
        self.next_residual = None
        self.correction = None
        self.trial = None
        self.residual_delta = None
        self.inverse_residual_delta = None
        self.scaled_update = None
        self.history_correction = None
        self.add_update = None
        self.difference_update = None
        self.size = -1

        self.inner_product = lambda left, right: block_inner_product(
            inner_product,
            left,
            right,
        )
        self.history = ResolventBroydenHistory(
            self.allocator,
            self.inner_product,
            depth,
        )


        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_monitorable = self.call_specialized
        else:
            self.call_monitorable = self.call_inline
        self.redirect_call = self.call_monitorable

    def prepare_specialized_kernels(
        self,
        specialist: ResolventSpecialist[Translation],
    ) -> None:
        # Step 3/6: add block corrections.
        self.add_update = specialist.provide(ResolventStencilBlock((1.0, 1.0)))
        # Step 4/5: form block differences.
        self.difference_update = specialist.provide(
            ResolventStencilBlock((1.0, -1.0))
        )

    def prepare_buffers(self, delta: Block[Translation]) -> None:
        size = len(delta)
        if self.size == size:
            return

        self.size = size
        self.residual_buffer = self.allocator.allocate_like(delta)
        self.next_residual = self.allocator.allocate_like(delta)
        self.correction = self.allocator.allocate_like(delta)
        self.trial = self.allocator.allocate_like(delta)
        self.residual_delta = self.allocator.allocate_like(delta)
        self.inverse_residual_delta = self.allocator.allocate_like(delta)
        self.scaled_update = self.allocator.allocate_like(delta)
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
        next_residual = cast(Block[Translation], self.next_residual)
        correction = cast(Block[Translation], self.correction)
        trial = cast(Block[Translation], self.trial)
        residual_delta = cast(Block[Translation], self.residual_delta)
        inverse_residual_delta = cast(Block[Translation], self.inverse_residual_delta)
        scaled_update = cast(Block[Translation], self.scaled_update)
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

            # 3. Apply the inverse approximation and negate it.
            self.history.apply_inverse(residual, history_correction, correction)
            correction.replace(-1.0 * correction)

            # 4. Trial the correction and compute the residual change.
            trial.replace(delta + correction)
            F(trial, next_residual)
            residual_delta.replace(next_residual - residual)

            # 5. Store a new inverse-Broyden secant pair when informative.
            denominator = self.inner_product(residual_delta, residual_delta)
            if denominator > 0.0:
                self.history.apply_inverse(
                    residual_delta,
                    history_correction,
                    inverse_residual_delta,
                )
                scaled_update.replace(correction - inverse_residual_delta)
                scaled_update.replace((1.0 / denominator) * scaled_update)
                self.history.append(scaled_update, residual_delta)

            # 6. Accept the trial iterate.
            delta.replace(trial)
            iteration_count += 1

        # 7. Recheck once after the final correction.
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
        next_residual = cast(Block[Translation], self.next_residual)
        correction = cast(Block[Translation], self.correction)
        trial = cast(Block[Translation], self.trial)
        residual_delta = cast(Block[Translation], self.residual_delta)
        inverse_residual_delta = cast(Block[Translation], self.inverse_residual_delta)
        scaled_update = cast(Block[Translation], self.scaled_update)
        history_correction = cast(Block[Translation], self.history_correction)
        add_update = self.add_update
        difference_update = self.difference_update
        assert add_update is not None
        assert difference_update is not None

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

            # 3. Apply the inverse approximation and negate it.
            self.history.apply_inverse(residual, history_correction, correction)
            correction.replace(-1.0 * correction)

            # 4. Trial the correction and compute the residual change.
            add_update(1.0, trial, delta, correction)
            F(trial, next_residual)
            difference_update(1.0, residual_delta, next_residual, residual)

            # 5. Store a new inverse-Broyden secant pair when informative.
            denominator = self.inner_product(residual_delta, residual_delta)
            if denominator > 0.0:
                self.history.apply_inverse(
                    residual_delta,
                    history_correction,
                    inverse_residual_delta,
                )
                difference_update(
                    1.0,
                    scaled_update,
                    correction,
                    inverse_residual_delta,
                )
                scaled_update.replace((1.0 / denominator) * scaled_update)
                self.history.append(scaled_update, residual_delta)

            # 6. Accept the trial iterate.
            delta.replace(trial)
            iteration_count += 1

        # 7. Recheck once after the final correction.
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

    def __call__(self, problem, delta):
        return self.redirect_call(problem, delta)

__all__ = ["ResolventBroyden", "ResolventBroydenHistory"]
