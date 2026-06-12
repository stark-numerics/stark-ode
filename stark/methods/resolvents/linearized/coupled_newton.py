from __future__ import annotations

from stark.core import Configuration
from stark.methods.resolvents.configuration import ResolventConfiguration
"""Newton-backed resolvent for fully coupled implicit RK stage systems."""

from typing import TYPE_CHECKING, Any, cast

from stark.core.block import Block, BlockAllocator, BlockOperatorDiagonal
from stark.core.contracts import (
    Accelerator,
    BlockOperatorLike,
    Inverter,
    InverterOutputMode,
    Linearizer,
    Translation,
    Allocator,
)
from stark.engines.accelerators import AcceleratorNone
from stark.methods.resolvents.method.descriptor import ResolventDescriptor
from stark.methods.resolvents.method.errors import ResolventError
from stark.methods.resolvents.monitoring.monitor import MonitorResolventLike
from stark.methods.resolvents.monitoring.decorators import with_resolvent_monitoring
from stark.methods.resolvents.display.decorators import with_resolvent_display
from stark.methods.resolvents.requests.inverter import ResolventInverterRequest
from stark.methods.resolvents.requests.resolvent import ResolventRequestCoupled
from stark.methods.resolvents.equations.implicit import ResolventImplicitEquationCoupled
from stark.methods.resolvents.specialization.specialist import ResolventSpecialist
from stark.methods.resolvents.specialization.stencil import ResolventStencilBlock
from stark.methods.resolvents.method.safety import ResolventSafety, ResolventSafetyDefault


# Optional extension: adds human-readable resolvent metadata and formatting helpers.
# Provides: short_name, __repr__, and __str__.
@with_resolvent_display
# Optional extension: records resolvent monitor events.
# Provides: assign_monitor, unassign_monitor, and record_solve.
@with_resolvent_monitoring
class ResolventCoupledNewton:
    """Newton iteration for coupled implicit RK stage systems.

    Algorithm sketch:

        1. Compute the coupled residual F(delta).
        2. Accept if ||F(delta)|| is within Tolerance.
        3. Build the coupled differential DF(delta).
        4. Solve DF(delta) correction = -F(delta).
        5. Apply delta <- delta + correction.
        6. Recheck once after the final correction.
    """

    __slots__ = (
        "_monitor",
        "accelerator",
        "alpha",
        "allocator",
        "call_step",
        "correction",
        "inverter",
        "newton_update",
        "operator",
        "max_iterations",
        "redirect_call",
        "equation",
        "residual_buffer",
        "rhs_buffer",
        "safety",
        "solve_correction",
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
        inverter: Inverter[Translation],
        configuration: ResolventConfiguration | None = None,
        safety: ResolventSafety | None = None,
        accelerator: Accelerator | None = None,
        specialist: ResolventSpecialist[Translation] | None = None,
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
        self.inverter = inverter
        self.solve_correction = (
            self.solve_correction_overwrite
            if getattr(inverter, "output_mode", InverterOutputMode.improve)
            is InverterOutputMode.overwrite
            else self.solve_correction_improve
        )

        self.accelerator = accelerator if accelerator is not None else AcceleratorNone()
        self.equation = ResolventImplicitEquationCoupled(
            "ResolventCoupledNewton",
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
            self.call_step = self.call_specialized
        else:
            self.call_step = self.call_inline
        self.redirect_call = self.call_step

    def prepare_specialized_kernels(
        self,
        specialist: ResolventSpecialist[Translation],
    ) -> None:
        # Step 5: Newton update delta <- delta + correction.
        self.newton_update = specialist.provide(
            ResolventStencilBlock((1.0, 1.0))
        )

    def solve_correction_improve(
        self,
        operator: BlockOperatorLike[Translation],
        rhs: Block[Translation],
        correction: Block[Translation],
    ) -> None:
        correction.replace(0.0 * correction)
        request = ResolventInverterRequest(operator=operator, residual=rhs)
        self.inverter(request, correction)

    def solve_correction_overwrite(
        self,
        operator: BlockOperatorLike[Translation],
        rhs: Block[Translation],
        correction: Block[Translation],
    ) -> None:
        request = ResolventInverterRequest(operator=operator, residual=rhs)
        self.inverter(request, correction)

    def prepare_buffers(self, delta: Block[Translation]) -> None:
        size = len(delta)
        if self.size == size:
            return

        self.size = size
        self.correction = self.allocator.allocate_like(delta)
        self.residual_buffer = self.allocator.allocate_like(delta)
        self.rhs_buffer = self.allocator.allocate_like(delta)
        self.operator = BlockOperatorDiagonal(None for _ in range(size))

    def operator_for(self, delta: Block[Translation]) -> BlockOperatorDiagonal[Translation]:
        """Return the operator object that will receive the coupled Jacobian.

        A coupled implicit equation owns a block operator when the Jacobian action
        is naturally expressed as one coupled matrix action. Other residual
        workers can fall back to the ordinary per-entry BlockOperatorDiagonal buffer.
        """

        residual_owned_operator = self.equation.block_operator
        if residual_owned_operator is not None:
            return residual_owned_operator

        self.prepare_buffers(delta)
        return cast(BlockOperatorDiagonal[Translation], self.operator)

    def call_inline(
        self,
        problem: ResolventRequestCoupled,
        delta: Block[Translation],
    ) -> Block[Translation]:
        self.alpha = problem.step
        self.prepare_buffers(delta)

        equation = self.equation.prepare(problem)
        correction = cast(Block[Translation], self.correction)
        residual = cast(Block[Translation], self.residual_buffer)
        rhs = cast(Block[Translation], self.rhs_buffer)

        block_size = len(delta)
        iteration_count = 0

        for _ in range(self.max_iterations):
            # 1. Compute the coupled residual F(delta).
            equation(delta, residual)

            # 2. Accept if ||F(delta)|| is within Tolerance.
            error = residual.norm()
            scale = delta.norm()
            if self.tolerance.accepts(error, scale):
                self.record_solve(block_size, iteration_count, error, scale, True)
                return delta

            # 3. Build the coupled differential DF(delta).
            operator = self.operator_for(delta)
            operator.reset()
            equation.differential(delta, operator)

            # 4. Solve DF(delta) correction = -F(delta).
            rhs.replace(-1.0 * residual)
            self.solve_correction(operator, rhs, correction)

            # 5. Newton update: delta <- delta + correction.
            delta += correction
            iteration_count += 1

        # 6. Recheck once after the final correction.
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
        self.prepare_buffers(delta)

        equation = self.equation.prepare(problem)
        correction = cast(Block[Translation], self.correction)
        residual = cast(Block[Translation], self.residual_buffer)
        rhs = cast(Block[Translation], self.rhs_buffer)
        newton_update = self.newton_update
        assert newton_update is not None

        block_size = len(delta)
        iteration_count = 0

        for _ in range(self.max_iterations):
            # 1. Compute the coupled residual F(delta).
            equation(delta, residual)

            # 2. Accept if ||F(delta)|| is within Tolerance.
            error = residual.norm()
            scale = delta.norm()
            if self.tolerance.accepts(error, scale):
                self.record_solve(block_size, iteration_count, error, scale, True)
                return delta

            # 3. Build the coupled differential DF(delta).
            operator = self.operator_for(delta)
            operator.reset()
            equation.differential(delta, operator)

            # 4. Solve DF(delta) correction = -F(delta).
            rhs.replace(-1.0 * residual)
            self.solve_correction(operator, rhs, correction)

            # 5. Newton update: delta <- delta + correction.
            newton_update(1.0, delta, correction, delta)
            iteration_count += 1

        # 6. Recheck once after the final correction.
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

__all__ = ["ResolventCoupledNewton"]
