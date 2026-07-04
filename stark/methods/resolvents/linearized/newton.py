"""Newton-backed resolvent for one-stage shifted implicit solves."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic

from stark.core import Configuration
from stark.core.block import Block, BlockOperatorDiagonal
from stark.core.contracts import (
    Accelerator,
    Inverter,
    InverterOutputMode,
    LinearizerLike,
    StateType,
    TranslationType,
    Allocator,
)
from stark.engines.shared.accelerators import AcceleratorNone
from stark.methods.resolvents.configuration import ResolventConfiguration
from stark.methods.resolvents.method.descriptor import ResolventDescriptor
from stark.methods.resolvents.method.errors import ResolventError
from stark.methods.resolvents.monitoring.monitor import MonitorResolventLike
from stark.methods.resolvents.monitoring.decorators import with_resolvent_monitoring
from stark.methods.resolvents.display.decorators import with_resolvent_display
from stark.methods.resolvents.requests.inverter import ResolventInverterRequest
from stark.methods.resolvents.requests.resolvent import ResolventRequest
from stark.methods.resolvents.equations.implicit import ResolventImplicitEquation
from stark.methods.resolvents.specialization.specialist import ResolventSpecialist
from stark.methods.resolvents.specialization.stencil import ResolventStencilBlock
from stark.methods.resolvents.method.safety import ResolventSafety, ResolventSafetyDefault


# Optional extension: adds human-readable resolvent metadata and formatting helpers.
# Provides: short_name, __repr__, and __str__.
@with_resolvent_display
# Optional extension: records resolvent monitor events.
# Provides: assign_monitor, unassign_monitor, and record_solve.
@with_resolvent_monitoring
class ResolventNewton(Generic[StateType, TranslationType]):
    """Newton iteration for one-stage shifted implicit residuals.

    Residual equation:

        equation(delta) = delta - rhs - alpha * f(t, origin + delta)

    Algorithm sketch:

        1. Compute F(delta).
        2. Accept if ||F(delta)|| is within Tolerance.
        3. Build the differential DF(delta).
        4. Solve DF(delta) correction = -F(delta).
        5. Apply the Newton update delta <- delta + correction.
        6. Recheck once after the final correction.
    """

    __slots__ = (
        "_monitor",
        "accelerator",
        "alpha",
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
        allocator: Allocator[StateType, TranslationType],
        linearizer: LinearizerLike[StateType, TranslationType],
        inverter: Inverter[TranslationType],
        configuration: ResolventConfiguration | None = None,
        safety: ResolventSafety | None = None,
        accelerator: Accelerator | None = None,
        specialist: ResolventSpecialist[TranslationType] | None = None,
        tableau: Any | None = None,
    ) -> None:
        self.tableau = tableau
        self.safety = safety if safety is not None else ResolventSafetyDefault()
        self.alpha = 0.0
        self._monitor = None

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
        self.equation = ResolventImplicitEquation[StateType, TranslationType](
            "ResolventNewton",
            allocator,
            linearizer=linearizer,
            accelerator=self.accelerator,
        )
        self.correction = Block[TranslationType]([allocator.allocate_translation()])
        self.residual_buffer = Block[TranslationType]([allocator.allocate_translation()])
        self.rhs_buffer = Block[TranslationType]([allocator.allocate_translation()])
        self.operator = BlockOperatorDiagonal[TranslationType]([None])
        self.newton_update = None

        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_step = self.call_specialized
        else:
            self.call_step = self.call_inline
        self.redirect_call = self.call_step

    def prepare_specialized_kernels(
        self,
        specialist: ResolventSpecialist[TranslationType],
    ) -> None:
        # Step 5: Newton update delta <- delta + correction.
        self.newton_update = specialist.provide(
            ResolventStencilBlock((1.0, 1.0))
        )

    def solve_correction_improve(
        self,
        operator: BlockOperatorDiagonal[TranslationType],
        rhs: Block[TranslationType],
        correction: Block[TranslationType],
    ) -> None:
        correction.replace(0.0 * correction)
        request = ResolventInverterRequest(operator=operator, residual=rhs)
        self.inverter(request, correction)

    def solve_correction_overwrite(
        self,
        operator: BlockOperatorDiagonal[TranslationType],
        rhs: Block[TranslationType],
        correction: Block[TranslationType],
    ) -> None:
        request = ResolventInverterRequest(operator=operator, residual=rhs)
        self.inverter(request, correction)

    def call_inline(
        self,
        problem: ResolventRequest[StateType, TranslationType],
        delta: Block[TranslationType],
    ) -> Block[TranslationType]:
        self.alpha = problem.alpha

        equation = self.equation.prepare(problem)
        correction = self.correction
        residual = self.residual_buffer
        rhs = self.rhs_buffer
        operator = self.operator

        block_size = 1
        iteration_count = 0

        for _ in range(self.max_iterations):
            # 1. Compute F(delta).
            equation(delta, residual)

            # 2. Accept if ||F(delta)|| is within Tolerance.
            error = residual[0].norm()
            scale = delta[0].norm()
            if self.tolerance.accepts(error, scale):
                self.record_solve(block_size, iteration_count, error, scale, True)
                return delta

            # 3. Build the differential DF(delta).
            equation.differential(delta, operator)

            # 4. Solve DF(delta) correction = -F(delta).
            rhs[0] = equation.scale(-1.0, residual[0], rhs[0])
            self.solve_correction(operator, rhs, correction)

            # 5. Newton update: delta <- delta + correction.
            delta[0] = equation.combine2(1.0, delta[0], 1.0, correction[0], delta[0])
            iteration_count += 1

        # 6. Recheck once after the final correction.
        equation(delta, residual)

        error = residual[0].norm()
        scale = delta[0].norm()
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
        problem: ResolventRequest[StateType, TranslationType],
        delta: Block[TranslationType],
    ) -> Block[TranslationType]:
        self.alpha = problem.alpha

        equation = self.equation.prepare(problem)
        correction = self.correction
        residual = self.residual_buffer
        rhs = self.rhs_buffer
        operator = self.operator
        newton_update = self.newton_update

        block_size = 1
        iteration_count = 0

        for _ in range(self.max_iterations):
            # 1. Compute F(delta).
            equation(delta, residual)

            # 2. Accept if ||F(delta)|| is within Tolerance.
            error = residual[0].norm()
            scale = delta[0].norm()
            if self.tolerance.accepts(error, scale):
                self.record_solve(block_size, iteration_count, error, scale, True)
                return delta

            # 3. Build the differential DF(delta).
            equation.differential(delta, operator)

            # 4. Solve DF(delta) correction = -F(delta).
            rhs[0] = equation.scale(-1.0, residual[0], rhs[0])
            self.solve_correction(operator, rhs, correction)

            # 5. Newton update: delta <- delta + correction.
            newton_update(1.0, delta, correction, delta)  # type: ignore[operator]
            iteration_count += 1

        # 6. Recheck once after the final correction.
        equation(delta, residual)

        error = residual[0].norm()
        scale = delta[0].norm()
        if self.tolerance.accepts(error, scale):
            self.record_solve(block_size, iteration_count, error, scale, True)
            return delta

        self.record_solve(block_size, iteration_count, error, scale, False)
        raise ResolventError(
            f"{type(self).__name__} failed to resolve the residual within "
            f"{self.max_iterations} iterations (error={error:g})."
        )

    def __call__(
        self,
        problem: ResolventRequest[StateType, TranslationType],
        delta: Block[TranslationType],
    ) -> Block[TranslationType]:
        return self.redirect_call(problem, delta)

__all__ = ["ResolventNewton"]
