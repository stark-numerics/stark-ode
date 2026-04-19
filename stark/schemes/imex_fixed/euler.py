from __future__ import annotations

from stark.schemes.tableau import ButcherTableau, ImExButcherTableau
from stark.contracts import ImExDerivative, IntervalLike, Resolvent, State, Workbench
from stark.execution.regulator import Regulator
from stark.resolvents.support.guard import ResolventTableauGuard
from stark.schemes.descriptor import SchemeDescriptor
from stark.machinery.stage_solve.workers import ImExStepper
from stark.schemes.base import SchemeBaseImExFixed
from stark.execution.executor import Executor


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
    """
    First-order IMEX Euler with explicit and implicit derivative splits.

    This method advances one explicit stage at the current state and one
    implicit stage at the end of the step. It is a good baseline split method:
    cheap, structurally simple, and easy to customize around problem-specific
    implicit solves, but only first-order accurate.

    Further reading: Ascher, Ruuth, and Wetton, SIAM Journal on Numerical
    Analysis 32(3), 1995.
    """

    __slots__ = ("resolvent", "stepper", "tableau_guard", "workspace")

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
        self.stepper = ImExStepper(derivative, self.workspace, self.resolvent, self.tableau)


    def __call__(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        del executor
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        dt = interval.step if interval.step <= remaining else remaining
        delta_high, _error, _high_norm, _error_norm = self.stepper.step(interval, state, dt)
        self.workspace.apply_delta(delta_high, state)
        return dt


__all__ = ["IMEX_EULER_TABLEAU", "SchemeIMEXEuler"]













