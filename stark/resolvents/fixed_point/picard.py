from __future__ import annotations

"""Picard-backed resolvent for one-stage shifted implicit solves."""

from typing import TYPE_CHECKING, Any, cast

from stark.block import Block, BlockAllocator
from stark.contracts import AcceleratorLike, Translation, Workbench
from stark.execution.safety import Safety
from stark.execution.tolerance import Tolerance
from stark.resolvents.support import (
    MonitorResolventLike,
    ResolventError,
    ResolventPolicy,
    ResolventSpecialist,
    ResolventStageProblem,
    ResolventStageResidual,
    ResolventStencilBlock,
    initialise_resolvent_runtime,
    refresh_resolvent_call,
    with_resolvent_call_methods,
    with_resolvent_display_methods,
    with_resolvent_monitoring_methods,
)
from stark.resolvents.support.descriptor import ResolventDescriptor
from stark.resolvents.support.tolerance import ResolventTolerance


@with_resolvent_display_methods
@with_resolvent_call_methods
@with_resolvent_monitoring_methods
class ResolventPicard:
    """Picard iteration for one-stage shifted implicit residuals.

    Residual equation:

        F(delta) = delta - rhs - alpha * f(t, origin + delta)

    Algorithm sketch:

        1. Start from the current stage increment delta.
        2. Compute F(delta).
        3. Accept if ||F(delta)|| is within tolerance.
        4. Otherwise apply the Picard correction delta <- delta - F(delta).
        5. Recheck once after the final correction.
    """

    __slots__ = (
        "_monitor",
        "accelerator",
        "alpha",
        "allocator",
        "call_pure",
        "picard_update",
        "policy",
        "redirect_call",
        "residual",
        "residual_buffer",
        "safety",
        "size",
        "tableau",
        "tolerance",
    )

    descriptor = ResolventDescriptor("Picard", "Picard Iteration")

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
        workbench: Workbench,
        tolerance: Tolerance | None = None,
        policy: ResolventPolicy | None = None,
        safety: Safety | None = None,
        accelerator: AcceleratorLike | None = None,
        specialist: ResolventSpecialist[Translation] | None = None,
        tableau: Any | None = None,
    ) -> None:
        self.tableau = tableau
        initialise_resolvent_runtime(self, safety, accelerator)

        self.allocator = BlockAllocator(workbench)
        self.tolerance = (
            tolerance
            if tolerance is not None
            else ResolventTolerance(atol=1.0e-9, rtol=1.0e-9)
        )
        self.policy = policy if policy is not None else ResolventPolicy()
        self.residual = ResolventStageResidual(
            "ResolventPicard",
            workbench,
            accelerator=self.accelerator,
        )
        self.residual_buffer = None
        self.picard_update = None
        self.size = -1

        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_pure = self.call_specialized
            refresh_resolvent_call(self)

    def prepare_specialized_kernels(
        self,
        specialist: ResolventSpecialist[Translation],
    ) -> None:
        # Step 4: Picard correction delta <- delta - F(delta).
        self.picard_update = specialist.provide(
            ResolventStencilBlock((1.0, -1.0))
        )

    def residual_buffer_for(
        self,
        delta: Block[Translation],
    ) -> Block[Translation]:
        size = len(delta)
        if self.size != size:
            self.size = size
            self.residual_buffer = self.allocator.allocate_like(delta)

        return cast(Block[Translation], self.residual_buffer)

    def call_inline(
        self,
        problem: ResolventStageProblem,
        delta: Block[Translation],
    ) -> Block[Translation]:
        if self.policy.max_iterations < 1:
            raise ValueError("ResolventPolicy.max_iterations must be at least 1.")

        self.alpha = problem.alpha
        F = self.residual
        F.configure(problem)
        residual = self.residual_buffer_for(delta)

        block_size = len(delta)
        iteration_count = 0

        for _ in range(self.policy.max_iterations):
            # 2. Compute F(delta).
            F(delta, residual)

            # 3. Accept if ||F(delta)|| is within tolerance.
            error = residual.norm()
            scale = delta.norm()
            if self.tolerance.accepts(error, scale):
                self.record_solve(block_size, iteration_count, error, scale, True)
                return delta

            # 4. Picard correction: delta <- delta - F(delta).
            delta -= residual
            iteration_count += 1

        # 5. Recheck once after the final correction.
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
        F = self.residual
        F.configure(problem)
        residual = self.residual_buffer_for(delta)
        picard_update = self.picard_update
        assert picard_update is not None

        block_size = len(delta)
        iteration_count = 0

        for _ in range(self.policy.max_iterations):
            # 2. Compute F(delta).
            F(delta, residual)

            # 3. Accept if ||F(delta)|| is within tolerance.
            error = residual.norm()
            scale = delta.norm()
            if self.tolerance.accepts(error, scale):
                self.record_solve(block_size, iteration_count, error, scale, True)
                return delta

            # 4. Picard correction: delta <- delta - F(delta).
            picard_update(1.0, delta, delta, residual)
            iteration_count += 1

        # 5. Recheck once after the final correction.
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


__all__ = ["ResolventPicard"]
