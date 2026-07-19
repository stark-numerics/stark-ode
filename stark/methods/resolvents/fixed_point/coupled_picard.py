from __future__ import annotations

from stark.core import Configuration
from stark.methods.resolvents.configuration import ResolventConfiguration
"""Picard-backed resolvent for fully coupled implicit RK stage systems."""

from typing import TYPE_CHECKING, Any, cast

from stark.core.block import Block, BlockAllocator
from stark.core.contracts import Accelerator, Translation, AllocatorLike
from stark.engines.accelerators import AcceleratorNone
from stark.methods.resolvents.method.descriptor import ResolventDescriptor
from stark.methods.resolvents.method.errors import ResolventError
from stark.methods.resolvents.monitoring.monitor import MonitorResolventLike
from stark.methods.resolvents.monitoring.decorators import with_resolvent_monitoring
from stark.methods.resolvents.display.decorators import with_resolvent_display
from stark.methods.resolvents.requests.resolvent import ResolventRequestCoupled
from stark.methods.resolvents.equations.implicit import ResolventImplicitEquationCoupled
from stark.methods.resolvents.linear_fixed_generation.linear_fixed import ResolventLinearFixed
from stark.methods.resolvents.linear_fixed_generation.stencil import ResolventStencilBlock
from stark.methods.resolvents.method.safety import ResolventSafety, ResolventSafetyDefault


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
        3. Accept if ||F(delta)|| is within Tolerance.
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
        allocator: AllocatorLike,
        configuration: ResolventConfiguration | None = None,
        safety: ResolventSafety | None = None,
        accelerator: Accelerator | None = None,
        linear_fixed: ResolventLinearFixed[Translation] | None = None,
        tableau: Any | None = None,
    ) -> None:
        self.tableau = tableau
        self.safety = safety if safety is not None else ResolventSafetyDefault()
        self.alpha = 0.0
        self._monitor = None


        self.allocator = BlockAllocator(allocator)
        configuration = configuration if configuration is not None else Configuration()
        self.tolerance = configuration.resolvent_tolerance
        self.max_iterations = configuration.resolvent_maximum_steps

        self.accelerator = accelerator if accelerator is not None else AcceleratorNone()
        self.equation = ResolventImplicitEquationCoupled(
            "ResolventCoupledPicard",
            allocator,
            accelerator=self.accelerator,
        )
        
        self.residual_buffer = None
        self.picard_update = None
        self.size = -1
        
        if linear_fixed is not None:
            self.prepare_specialized_kernels(linear_fixed)
            self.call_step = self.call_specialized
        else:
            self.call_step = self.call_inline
        self.redirect_call = self.call_step

    def prepare_specialized_kernels(
        self,
        linear_fixed: ResolventLinearFixed[Translation],
    ) -> None:
        # Step 4: coupled Picard correction delta <- delta - F(delta).
        self.picard_update = linear_fixed(
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

            # 3. Accept if ||F(delta)|| is within Tolerance.
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

            # 3. Accept if ||F(delta)|| is within Tolerance.
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
