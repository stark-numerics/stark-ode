"""Chord-backed resolvent for one-stage shifted implicit solves."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from stark.core import Configuration
from stark.core.block import Block, BlockOperatorDiagonal
from stark.core.contracts import (
    Accelerator,
    Allocator,
    Inverter,
    InverterOutputMode,
    LinearizerLike,
    Translation,
)
from stark.engines.accelerators import AcceleratorNone
from stark.methods.resolvents.configuration import ResolventConfiguration, ResolventConfigurationDefault
from stark.methods.resolvents.display.decorators import with_resolvent_display
from stark.methods.resolvents.equations.implicit import ResolventImplicitEquation
from stark.methods.resolvents.method.descriptor import ResolventDescriptor
from stark.methods.resolvents.method.errors import ResolventError
from stark.methods.resolvents.method.safety import ResolventSafety, ResolventSafetyDefault
from stark.methods.resolvents.monitoring.decorators import with_resolvent_monitoring
from stark.methods.resolvents.monitoring.monitor import MonitorResolventLike
from stark.methods.resolvents.requests.inverter import ResolventInverterRequest
from stark.methods.resolvents.requests.resolvent import ResolventRequest
from stark.methods.resolvents.specialization.specialist import ResolventSpecialist
from stark.methods.resolvents.specialization.stencil import ResolventStencilBlock


@with_resolvent_display
@with_resolvent_monitoring
class ResolventChord:
    """Chord iteration for one-stage shifted implicit residuals.

    Residual equation:

        equation(delta) = delta - rhs - alpha * f(t, origin + delta)

    Chord iteration is Newton's close cousin. Newton rebuilds the differential
    ``DF(delta)`` after every correction. Chord builds one differential for the
    current stage solve and reuses it while iterating:

        DF(delta_initial) correction = -F(delta_current)

    This is useful when constructing or inverting the linearized operator is a
    meaningful part of the cost. It is also the right comparison point for
    solver stacks that use chord-like implicit stage root finders. The tradeoff
    is that each correction is less locally informed than Newton's, so Chord may
    need more correction iterations or fail earlier on difficult nonlinear
    stages. The equation keeps separate residual and linearization scratch
    states so residual evaluations cannot mutate closure-backed Jacobian state
    after the differential has been frozen.

    Algorithm sketch:

        1. Compute F(delta).
        2. Accept if ||F(delta)|| is within Tolerance.
        3. Build DF(delta) once for this stage solve.
        4. Repeatedly solve DF(initial) correction = -F(delta).
        5. Apply delta <- delta + correction and recompute F(delta).
        6. Raise a recoverable resolvent error if the configured iteration
           budget is exhausted, so adaptive schemes can reject and retry with a
           smaller step.
    """

    __slots__ = (
        "_monitor",
        "accelerator",
        "call_step",
        "correction",
        "equation",
        "inverter",
        "max_iterations",
        "operator",
        "redirect_call",
        "residual_buffer",
        "rhs_buffer",
        "safety",
        "solve_correction",
        "tableau",
        "tolerance",
        "update",
    )

    descriptor = ResolventDescriptor("Chord", "Chord Iteration")

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
        linearizer: LinearizerLike,
        inverter: Inverter[Translation],
        configuration: ResolventConfiguration | None = None,
        safety: ResolventSafety | None = None,
        accelerator: Accelerator | None = None,
        specialist: ResolventSpecialist[Translation] | None = None,
        tableau: Any | None = None,
    ) -> None:
        self.tableau = tableau
        self.safety = safety if safety is not None else ResolventSafetyDefault()
        self._monitor = None

        configuration = configuration if configuration is not None else ResolventConfigurationDefault()
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
        self.equation = ResolventImplicitEquation(
            "ResolventChord",
            allocator,
            linearizer=linearizer,
            accelerator=self.accelerator,
        )
        self.correction = Block([allocator.allocate_translation()])
        self.residual_buffer = Block([allocator.allocate_translation()])
        self.rhs_buffer = Block([allocator.allocate_translation()])
        self.operator = BlockOperatorDiagonal([None])
        self.update = None

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
        self.update = specialist.provide(ResolventStencilBlock((1.0, 1.0)))

    def solve_correction_improve(
        self,
        operator: BlockOperatorDiagonal[Translation],
        rhs: Block[Translation],
        correction: Block[Translation],
    ) -> None:
        correction.replace(0.0 * correction)
        request = ResolventInverterRequest(operator=operator, residual=rhs)
        self.inverter(request, correction)

    def solve_correction_overwrite(
        self,
        operator: BlockOperatorDiagonal[Translation],
        rhs: Block[Translation],
        correction: Block[Translation],
    ) -> None:
        request = ResolventInverterRequest(operator=operator, residual=rhs)
        self.inverter(request, correction)

    def call_inline(
        self,
        problem: ResolventRequest,
        delta: Block[Translation],
    ) -> Block[Translation]:
        equation = self.equation.prepare(problem)
        correction = self.correction
        residual = self.residual_buffer
        rhs = self.rhs_buffer
        operator = self.operator

        block_size = 1
        iteration_count = 0

        equation(delta, residual)
        error = residual[0].norm()
        scale = delta[0].norm()
        if self.tolerance.accepts(error, scale):
            self.record_solve(block_size, iteration_count, error, scale, True)
            return delta

        equation.differential(delta, operator)

        for _ in range(self.max_iterations):
            rhs[0] = equation.scale(-1.0, residual[0], rhs[0])
            self.solve_correction(operator, rhs, correction)
            delta[0] = equation.combine2(1.0, delta[0], 1.0, correction[0], delta[0])
            iteration_count += 1

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
        problem: ResolventRequest,
        delta: Block[Translation],
    ) -> Block[Translation]:
        equation = self.equation.prepare(problem)
        correction = self.correction
        residual = self.residual_buffer
        rhs = self.rhs_buffer
        operator = self.operator
        update = self.update

        block_size = 1
        iteration_count = 0

        equation(delta, residual)
        error = residual[0].norm()
        scale = delta[0].norm()
        if self.tolerance.accepts(error, scale):
            self.record_solve(block_size, iteration_count, error, scale, True)
            return delta

        equation.differential(delta, operator)

        for _ in range(self.max_iterations):
            rhs[0] = equation.scale(-1.0, residual[0], rhs[0])
            self.solve_correction(operator, rhs, correction)
            update(1.0, delta, correction, delta)  # type: ignore[operator]
            iteration_count += 1

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

    def __call__(self, problem, delta):
        return self.redirect_call(problem, delta)


__all__ = ["ResolventChord"]
