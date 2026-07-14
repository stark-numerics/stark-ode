from __future__ import annotations

"""Very-chord resolvent for one-stage shifted implicit solves."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from stark.core import Configuration
from stark.core.block import Block, BlockOperatorDiagonal
from stark.core.contracts import (
    Accelerator,
    AllocatorLike,
    Inverter,
    InverterInstance,
    InverterOutputMode,
    LinearizerLike,
    Translation,
)
from stark.engines.accelerators import AcceleratorNone
from stark.methods.resolvents.configuration import ResolventConfiguration
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
class ResolventVeryChord:
    """Step-frozen chord iteration for shifted implicit residuals.

    Residual equation:

        equation(delta) = delta - rhs - alpha * f(t, origin + delta)

    Newton rebuilds the differential after every correction. Chord freezes the
    differential for one stage solve. Very-chord goes one level further for
    singly diagonally implicit methods: it freezes the differential for a step
    attempt and reuses that operator across all stages whose shifted intervals
    match the same tableau step start and alpha.

    This is an approximation. It is useful when materialising or inverting the
    linearized operator dominates the cost, but it is less locally informed
    than chord or Newton. The cache is conservative: if a request cannot be
    matched to the cached tableau step attempt, the differential is rebuilt.
    If the inverter provides `instance(operator)`, very-chord also reuses the
    operator-bound inverter instance. Without that capability it falls back to
    the ordinary inverter call shape. Without a tableau the resolvent behaves
    like ordinary chord.
    """

    __slots__ = (
        "_monitor",
        "accelerator",
        "call_step",
        "cached_alpha",
        "cached_solve",
        "cached_step_start",
        "correction",
        "diagonal",
        "equation",
        "inverter",
        "max_iterations",
        "operator",
        "redirect_call",
        "residual_buffer",
        "rhs_buffer",
        "safety",
        "solve_correction",
        "stage_abscissae",
        "tableau",
        "tolerance",
        "update",
    )

    descriptor = ResolventDescriptor("VeryChord", "Very Chord Iteration")

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
        self.equation = ResolventImplicitEquation(
            "ResolventVeryChord",
            allocator,
            linearizer=linearizer,
            accelerator=self.accelerator,
        )
        self.correction = Block([allocator.allocate_translation()])
        self.residual_buffer = Block([allocator.allocate_translation()])
        self.rhs_buffer = Block([allocator.allocate_translation()])
        self.operator = BlockOperatorDiagonal([None])
        self.update = None
        self.cached_alpha = None
        self.cached_step_start = None
        self.cached_solve: (
            Callable[[Block[Translation], Block[Translation]], None] | None
        ) = None
        self.diagonal, self.stage_abscissae = self._tableau_cache_shape(tableau)

        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_step = self.call_specialized
        else:
            self.call_step = self.call_inline
        self.redirect_call = self.call_step

    @staticmethod
    def _tableau_cache_shape(tableau: Any | None) -> tuple[float | None, tuple[float, ...]]:
        if tableau is None:
            return None, ()

        diagonal = None
        stage_abscissae = []
        for c_value, row in zip(tableau.c, tableau.a, strict=True):
            if len(row) < 2:
                continue
            row_diagonal = float(row[-1])
            if row_diagonal == 0.0:
                continue
            if diagonal is None:
                diagonal = row_diagonal
            stage_abscissae.append(float(c_value))

        return diagonal, tuple(stage_abscissae)

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

    def solve_cached_operator(
        self,
        rhs: Block[Translation],
        correction: Block[Translation],
    ) -> None:
        self.solve_correction(self.operator, rhs, correction)

    def inverter_instance(
        self,
    ) -> InverterInstance[Translation] | Callable[[Block[Translation], Block[Translation]], None]:
        instance = getattr(self.inverter, "instance", None)
        if instance is None:
            return self.solve_cached_operator
        return instance(self.operator)

    def cached_operator_matches(self, problem: ResolventRequest) -> bool:
        cached_alpha = self.cached_alpha
        cached_step_start = self.cached_step_start
        diagonal = self.diagonal
        if cached_alpha is None or cached_step_start is None or diagonal is None:
            return False
        if abs(problem.alpha - cached_alpha) > 1.0e-15 * max(1.0, abs(cached_alpha)):
            return False

        dt = problem.alpha / diagonal
        present = problem.interval.present
        tolerance = 1.0e-12 * max(1.0, abs(cached_step_start), abs(present), abs(dt))
        for c_value in self.stage_abscissae:
            if abs((present - c_value * dt) - cached_step_start) <= tolerance:
                return True
        return False

    def refresh_cached_operator(self, problem: ResolventRequest, delta: Block[Translation]) -> None:
        self.equation.differential(delta, self.operator)
        self.cached_solve = self.inverter_instance()
        diagonal = self.diagonal
        if diagonal is None:
            self.cached_alpha = None
            self.cached_step_start = None
            return

        dt = problem.alpha / diagonal
        first_stage = self.stage_abscissae[0]
        self.cached_alpha = problem.alpha
        self.cached_step_start = problem.interval.present - first_stage * dt

    def call_inline(
        self,
        problem: ResolventRequest,
        delta: Block[Translation],
    ) -> Block[Translation]:
        equation = self.equation.prepare(problem)
        correction = self.correction
        residual = self.residual_buffer
        rhs = self.rhs_buffer

        block_size = 1
        iteration_count = 0

        equation(delta, residual)
        error = residual[0].norm()
        scale = delta[0].norm()
        if self.tolerance.accepts(error, scale):
            self.record_solve(block_size, iteration_count, error, scale, True)
            return delta

        if not self.cached_operator_matches(problem):
            self.refresh_cached_operator(problem, delta)
        solve = self.cached_solve
        if solve is None:
            solve = self.solve_cached_operator

        for _ in range(self.max_iterations):
            rhs[0] = equation.scale(-1.0, residual[0], rhs[0])
            solve(rhs, correction)
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
        update = self.update

        block_size = 1
        iteration_count = 0

        equation(delta, residual)
        error = residual[0].norm()
        scale = delta[0].norm()
        if self.tolerance.accepts(error, scale):
            self.record_solve(block_size, iteration_count, error, scale, True)
            return delta

        if not self.cached_operator_matches(problem):
            self.refresh_cached_operator(problem, delta)
        solve = self.cached_solve
        if solve is None:
            solve = self.solve_cached_operator

        for _ in range(self.max_iterations):
            rhs[0] = equation.scale(-1.0, residual[0], rhs[0])
            solve(rhs, correction)
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


__all__ = ["ResolventVeryChord"]
