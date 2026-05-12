from __future__ import annotations

from stark.contracts import ImExDerivative, IntervalLike, Resolvent, State, Workbench
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator
from stark.machinery.stage_solve.workers import ImExStepper
from stark.resolvents.support.guard import ResolventTableauGuard
from stark.schemes.base import SchemeBaseImExFixed
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.tableau import ButcherTableau, ImExButcherTableau


IMEX_EULER_EXPLICIT = ButcherTableau(
    c=(0.0, 1.0),
    a=((), (1.0,)),
    b=(1.0, 0.0),
    order=1,
)

IMEX_EULER_IMPLICIT = ButcherTableau(
    c=(0.0, 1.0),
    a=((), (0.0, 1.0)),
    b=(0.0, 1.0),
    order=1,
)

IMEX_EULER_TABLEAU = ImExButcherTableau(
    explicit=IMEX_EULER_EXPLICIT,
    implicit=IMEX_EULER_IMPLICIT,
    short_name="IMEXEuler",
    full_name="IMEX Euler",
)


class SchemeIMEXEuler(SchemeBaseImExFixed):
    """First-order IMEX Euler with explicit and implicit derivative splits.

    IMEX Euler advances with an explicit contribution from the non-stiff part
    and an implicit contribution from the stiff part:

        x_{n+1} = x_n
                + dt f_explicit(t_n, x_n)
                + dt f_implicit(t_{n+1}, x_{n+1})

    The explicit derivative is evaluated at the current state. The implicit
    correction is delegated to the configured one-stage resolvent. The scheme
    file intentionally keeps that split visible: this is the simplest fixed-step
    example of STARK's additive explicit/implicit machinery.

    Further reading:
    Ascher, Ruuth, and Wetton, "Implicit-explicit methods for time-dependent
    partial differential equations", SIAM Journal on Numerical Analysis 32(3),
    1995.
    """

    __slots__ = (
        "call_pure",
        "redirect_call",
        "resolvent",
        "stepper",
        "tableau_guard",
        "workspace",
    )

    descriptor = SchemeDescriptor("IMEXEuler", "IMEX Euler")
    tableau = IMEX_EULER_TABLEAU

    def __init__(
        self,
        derivative: ImExDerivative,
        workbench: Workbench,
        resolvent: Resolvent,
        regulator: Regulator | None = None,
    ) -> None:
        del regulator

        super().__init__(derivative, workbench)

        self.tableau_guard = ResolventTableauGuard("IMEXEuler", self.tableau)

        if resolvent is None:
            raise TypeError("IMEXEuler requires an explicit resolvent.")

        self.resolvent = resolvent
        self.tableau_guard(self.resolvent)

        self.stepper = ImExStepper(
            derivative,
            self.workspace,
            self.resolvent,
            self.tableau,
        )

        self.call_pure = self.call_generic
        self.redirect_call = self.call_pure

    def __call__(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        return self.redirect_call(interval, state, executor)

    def call_generic(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        del executor

        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        dt = interval.step if interval.step <= remaining else remaining

        # `ImExStepper` keeps the explicit and implicit stage machinery
        # separated internally. At this call site the important public recipe is:
        # compute the IMEX step delta, apply it, then return the accepted dt.
        delta_high, _error, _high_norm, _error_norm = self.stepper.step(
            interval,
            state,
            dt,
        )
        self.workspace.apply_delta(delta_high, state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)

        return dt


__all__ = ["IMEX_EULER_TABLEAU", "SchemeIMEXEuler"]