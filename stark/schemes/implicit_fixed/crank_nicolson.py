from __future__ import annotations

from stark.accelerators.binding import DerivativeAccelerated
from stark.algebraist.classic import Algebraist, AlgebraistImplicitCombination
from stark.contracts import Block, Derivative, IntervalLike, Resolvent, State, Workbench
from stark.execution.executor import Executor
from stark.machinery.stage_solve.workers import ShiftedOneStageResolventStep
from stark.schemes.support.descriptor import SchemeDescriptor
from stark.schemes.support import (
    refresh_fixed_step_call,
    unbound_scheme_call,
    with_fixed_step_monitoring,
    with_implicit_stepper_methods,
    with_scheme_display,
)
from stark.schemes.support.tableau import ButcherTableau


CRANK_NICOLSON_TABLEAU = ButcherTableau(
    c=(0.0, 1.0),
    a=((), (0.5, 0.5)),
    b=(0.5, 0.5),
    order=2,
    short_name="CN",
    full_name="Crank-Nicolson",
)


@with_scheme_display
@with_fixed_step_monitoring
@with_implicit_stepper_methods
class SchemeCrankNicolson:
    """The fixed-step Crank-Nicolson / trapezoidal Runge-Kutta method.

    This method evaluates one explicit stage at the start of the step, then
    resolves a one-stage shifted implicit problem at the end of the step:

        delta = (dt / 2) f(t_n, x_n)
              + (dt / 2) f(t_n + dt, x_n + delta).

    It is a simple but important bridge case between purely explicit methods and
    fully implicit sequential DIRK methods.

    Further reading:
    https://en.wikipedia.org/wiki/Trapezoidal_rule_(differential_equations)
    """

    __slots__ = (
        "_monitor",
        "call_pure",
        "derivative",
        "k1",
        "known_block",
        "known_rhs_call",
        "redirect_call",
        "stepper",
    )

    descriptor = SchemeDescriptor("CN", "Crank-Nicolson")
    tableau = CRANK_NICOLSON_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        resolvent: Resolvent,
        *,
        algebraist: Algebraist | None = None,
    ) -> None:
        self.known_rhs_call = unbound_scheme_call
        self._monitor = None
        self.derivative = DerivativeAccelerated(derivative)
        self.stepper = ShiftedOneStageResolventStep(
            "Crank-Nicolson",
            self.tableau,
            derivative,
            workbench,
            resolvent,
        )

        workspace = self.stepper.workspace
        self.k1 = workbench.allocate_translation()
        self.known_block = Block([workspace.allocate_translation()])

        self.call_pure = self.call_inline
        refresh_fixed_step_call(self)

        if algebraist is not None:
            self.use_specialists(algebraist)

    def __call__(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        return self.redirect_call(interval, state, executor)

    def use_specialists(self, algebraist: Algebraist) -> None:
        calls = algebraist.bind_implicit_fixed_scheme(
            known_shifts=(
                AlgebraistImplicitCombination(
                    "known_rhs",
                    (0.5,),
                    step_scale=True,
                ),
            ),
        )
        self.known_rhs_call = calls.require_known_shift_call(0, type(self).__name__)
        self.call_pure = self.call_specialized
        refresh_fixed_step_call(self)

    def call_inline(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        del executor

        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        workspace = self.stepper.workspace
        derivative = self.derivative
        known_block = self.known_block
        k1 = self.k1

        dt = interval.step if interval.step <= remaining else remaining

        derivative(interval, state, k1)
        known_block.items[0] = workspace.scale(0.5 * dt, k1, known_block[0])

        delta = self.stepper.solve(
            interval,
            state,
            dt,
            alpha=0.5 * dt,
            stage_shift=dt,
            rhs=known_block,
        )
        workspace.apply_delta(delta, state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)

        return dt

    def call_specialized(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        del executor

        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        workspace = self.stepper.workspace
        derivative = self.derivative
        known_block = self.known_block
        known_rhs_call = self.known_rhs_call
        k1 = self.k1

        dt = interval.step if interval.step <= remaining else remaining

        derivative(interval, state, k1)
        known_block.items[0] = known_rhs_call(dt, k1, known_block[0])

        delta = self.stepper.solve(
            interval,
            state,
            dt,
            alpha=0.5 * dt,
            stage_shift=dt,
            rhs=known_block,
        )
        workspace.apply_delta(delta, state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)

        return dt


__all__ = ["CRANK_NICOLSON_TABLEAU", "SchemeCrankNicolson"]
