from __future__ import annotations

"""Picard-backed resolvent for fully coupled implicit RK stage systems."""

from typing import TYPE_CHECKING, Any, cast

from stark.block import Block, BlockAllocator
from stark.contracts import AcceleratorLike, Translation, Allocator
from stark.accelerators import AcceleratorAbsent
from stark.executor.tolerance import ExecutorTolerance
from stark.resolvents.method.descriptor import ResolventDescriptor
from stark.resolvents.method.errors import ResolventError
from stark.resolvents.method.policy import ResolventPolicy
from stark.resolvents.monitoring.monitor import MonitorResolventLike
from stark.resolvents.monitoring.decorators import with_resolvent_monitoring
from stark.resolvents.display.decorators import with_resolvent_display
from stark.resolvents.requests.resolvent import ResolventRequestCoupled
from stark.resolvents.equations.implicit import ResolventImplicitEquationCoupled
from stark.resolvents.specialization.specialist import ResolventSpecialist
from stark.resolvents.specialization.stencil import ResolventStencilBlock
from stark.resolvents.method.tolerance import ResolventTolerance
from stark.resolvents.method.safety import ResolventSafety, ResolventSafetyDefault


# Optional extension: adds human-readable resolvent metadata and formatting helpers.
# Provides: short_name, __repr__, and __str__.
@with_resolvent_display
# Optional extension: records resolvent monitor events.
# Provides: assign_monitor, unassign_monitor, and record_solve.
@with_resolvent_monitoring
class ResolventCoupledPicard:
    """Picard iteration for fully coupled implicit RK stage systems.

    Algorithm sketch:

        1. Start from the current block of stage increments delta.
        2. Compute the coupled residual F(delta).
        3. Accept if ||F(delta)|| is within ExecutorTolerance.
        4. Otherwise apply delta <- delta - F(delta).
        5. Recheck once after the final correction.
    """

    __slots__ = (
        "_monitor",
        "accelerator",
        "alpha",
        "allocator",
        "call_step",
        "picard_update",
        "policy",
        "max_iterations",
        "redirect_call",
        "equation",
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
        allocator: Allocator,
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
        self.max_iterations = self.policy.max_iterations

        self.accelerator = accelerator if accelerator is not None else AcceleratorAbsent()
        self.equation = ResolventImplicitEquationCoupled(
            "ResolventCoupledPicard",
            allocator,
            accelerator=self.accelerator,
        )
        
        self.residual_buffer = None
        self.picard_update = None
        self.size = -1
        
        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_step = self.call_specialized
        else:
            self.call_step = self.call_inline
        self.redirect_call = self.call_step

    def prepare_specialized_kernels(
        self,
        specialist: ResolventSpecialist[Translation],
    ) -> None:
        # Step 4: coupled Picard correction delta <- delta - F(delta).
        self.picard_update = specialist.provide(
            ResolventStencilBlock((1.0, -1.0))
        )

    def residual_scratch(
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
        problem: ResolventRequestCoupled,
        delta: Block[Translation],
    ) -> Block[Translation]:
        self.alpha = problem.step
        equation = self.equation.prepare(problem)
        residual = self.residual_scratch(delta)

        block_size = len(delta)
        iteration_count = 0

        for _ in range(self.max_iterations):
            # 2. Compute the coupled residual F(delta).
            equation(delta, residual)

            # 3. Accept if ||F(delta)|| is within ExecutorTolerance.
            error = residual.norm()
            scale = delta.norm()
            if self.tolerance.accepts(error, scale):
                self.record_solve(block_size, iteration_count, error, scale, True)
                return delta

            # 4. Coupled Picard correction: delta <- delta - F(delta).
            delta -= residual
            iteration_count += 1

        # 5. Recheck once after the final correction.
        equation(delta, residual)

        error = residual.norm()
        scale = delta.norm()
        if self.tolerance.accepts(error, scale):
            self.record_solve(block_size, iteration_count, error, scale, True)
            return delta

        self.record_solve(block_size, iteration_count, error, scale, False)
        raise ResolventError(
            f"{type(self).__name__} failed to resolve the residual within "
            f"{self.max_iterations} iterations (error={error:g})."
        )

    def call_specialized(
        self,
        problem: ResolventRequestCoupled,
        delta: Block[Translation],
    ) -> Block[Translation]:
        self.alpha = problem.step
        equation = self.equation.prepare(problem)
        residual = self.residual_scratch(delta)
        picard_update = self.picard_update
        assert picard_update is not None

        block_size = len(delta)
        iteration_count = 0

        for _ in range(self.max_iterations):
            # 2. Compute the coupled residual F(delta).
            equation(delta, residual)

            # 3. Accept if ||F(delta)|| is within ExecutorTolerance.
            error = residual.norm()
            scale = delta.norm()
            if self.tolerance.accepts(error, scale):
                self.record_solve(block_size, iteration_count, error, scale, True)
                return delta

            # 4. Coupled Picard correction: delta <- delta - F(delta).
            picard_update(1.0, delta, delta, residual)
            iteration_count += 1

        # 5. Recheck once after the final correction.
        equation(delta, residual)

        error = residual.norm()
        scale = delta.norm()
        if self.tolerance.accepts(error, scale):
            self.record_solve(block_size, iteration_count, error, scale, True)
            return delta

        self.record_solve(block_size, iteration_count, error, scale, False)
        raise ResolventError(
            f"{type(self).__name__} failed to resolve the residual within "
            f"{self.max_iterations} iterations (error={error:g})."
        )

    def __call__(self, problem, delta):
        return self.redirect_call(problem, delta)

__all__ = ["ResolventCoupledPicard"]
