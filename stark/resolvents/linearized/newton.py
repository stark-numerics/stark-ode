from __future__ import annotations

"""Newton-backed resolvent for one-stage shifted implicit solves."""

from typing import TYPE_CHECKING, Any, cast

from stark.block import Block, BlockAllocator, BlockOperator
from stark.contracts import (
    AcceleratorLike,
    LegacyInverterLike,
    Linearizer,
    Translation,
    Allocator,
)
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
from stark.resolvents.support.tolerance import ResolventTolerance
from stark.resolvents.support.safety import ResolventSafety, ResolventSafetyDefault


@with_resolvent_display
@with_resolvent_monitoring
class ResolventNewton:
    """Newton iteration for one-stage shifted implicit residuals.

    Residual equation:

        F(delta) = delta - rhs - alpha * f(t, origin + delta)

    Algorithm sketch:

        1. Compute F(delta).
        2. Accept if ||F(delta)|| is within ExecutorTolerance.
        3. Build the differential DF(delta).
        4. Solve DF(delta) correction = -F(delta).
        5. Apply the Newton update delta <- delta + correction.
        6. Recheck once after the final correction.
    """

    __slots__ = (
        "_monitor",
        "accelerator",
        "alpha",
        "allocator",
        "call_monitorable",
        "correction",
        "inverter",
        "newton_update",
        "operator",
        "policy",
        "redirect_call",
        "residual",
        "residual_buffer",
        "rhs_buffer",
        "safety",
        "size",
        "tableau",
        "tolerance",
    )

    descriptor = ResolventDescriptor("Newton", "Newton Iteration")

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
        linearizer: Linearizer,
        inverter: LegacyInverterLike,
        ExecutorTolerance: ExecutorTolerance | None = None,
        policy: ResolventPolicy | None = None,
        safety: ResolventSafety | None = None,
        accelerator: AcceleratorLike | None = None,
        specialist: ResolventSpecialist[Translation] | None = None,
        tableau: Any | None = None,
    ) -> None:
        self.tableau = tableau
        self.safety = safety if safety is not None else ResolventSafetyDefault()
        self.alpha = 0.0
        self._monitor = None

        self.allocator = BlockAllocator(allocator)
        self.tolerance = (
            ExecutorTolerance
            if ExecutorTolerance is not None
            else ResolventTolerance(atol=1.0e-9, rtol=1.0e-9)
        )
        self.policy = policy if policy is not None else ResolventPolicy()
        self.inverter = inverter

        self.accelerator = accelerator if accelerator is not None else AcceleratorAbsent()
        self.residual = ResolventStageResidual(
            "ResolventNewton",
            allocator,
            linearizer=linearizer,
            accelerator=self.accelerator,
        )
        self.correction = None
        self.residual_buffer = None
        self.rhs_buffer = None
        self.operator = None
        self.newton_update = None
        self.size = -1

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
        # Step 5: Newton update delta <- delta + correction.
        self.newton_update = specialist.provide(
            ResolventStencilBlock((1.0, 1.0))
        )

    def prepare_buffers(self, delta: Block[Translation]) -> None:
        size = len(delta)
        if self.size == size:
            return

        self.size = size
        self.correction = self.allocator.allocate_like(delta)
        self.residual_buffer = self.allocator.allocate_like(delta)
        self.rhs_buffer = self.allocator.allocate_like(delta)
        self.operator = BlockOperator(None for _ in range(size))

    def call_inline(
        self,
        problem: ResolventStageProblem,
        delta: Block[Translation],
    ) -> Block[Translation]:
        if self.policy.max_iterations < 1:
            raise ValueError("ResolventPolicy.max_iterations must be at least 1.")

        self.alpha = problem.alpha
        self.prepare_buffers(delta)

        F = self.residual
        F.configure(problem)
        correction = cast(Block[Translation], self.correction)
        residual = cast(Block[Translation], self.residual_buffer)
        rhs = cast(Block[Translation], self.rhs_buffer)
        operator = cast(BlockOperator[Translation], self.operator)

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

            # 3. Build the differential DF(delta).
            operator.reset()
            F.differential(delta, operator)

            # 4. Solve DF(delta) correction = -F(delta).
            rhs.replace(-1.0 * residual)
            correction.replace(0.0 * correction)
            self.inverter.bind(operator)
            self.inverter(rhs, correction)

            # 5. Newton update: delta <- delta + correction.
            delta += correction
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

        F = self.residual
        F.configure(problem)
        correction = cast(Block[Translation], self.correction)
        residual = cast(Block[Translation], self.residual_buffer)
        rhs = cast(Block[Translation], self.rhs_buffer)
        operator = cast(BlockOperator[Translation], self.operator)
        newton_update = self.newton_update
        assert newton_update is not None

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

            # 3. Build the differential DF(delta).
            operator.reset()
            F.differential(delta, operator)

            # 4. Solve DF(delta) correction = -F(delta).
            rhs.replace(-1.0 * residual)
            correction.replace(0.0 * correction)
            self.inverter.bind(operator)
            self.inverter(rhs, correction)

            # 5. Newton update: delta <- delta + correction.
            newton_update(1.0, delta, delta, correction)
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

    def __call__(self, problem, delta):
        return self.redirect_call(problem, delta)

__all__ = ["ResolventNewton"]
