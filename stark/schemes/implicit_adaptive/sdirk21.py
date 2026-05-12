from __future__ import annotations

from math import sqrt

from stark.accelerators.binding import BoundDerivative
from stark.contracts import Derivative, IntervalLike, Resolvent, State, Workbench
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator
from stark.monitor import MonitorStep
from stark.resolvents.failure import ResolventError
from stark.machinery.stage_solve.workers import SequentialDIRKResolventStep
from stark.schemes.base import SchemeBaseImplicitAdaptive
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.tableau import ButcherTableau


SDIRK21_GAMMA = (2.0 - sqrt(2.0)) / 2.0

SDIRK21_B2 = (1.0 - 2.0 * SDIRK21_GAMMA) / (4.0 * SDIRK21_GAMMA)
SDIRK21_B1 = 1.0 - SDIRK21_B2 - SDIRK21_GAMMA

SDIRK21_BHAT2 = (
    SDIRK21_GAMMA
    * (
        -2.0
        + 7.0 * SDIRK21_GAMMA
        - 5.0 * SDIRK21_GAMMA * SDIRK21_GAMMA
        + 4.0 * SDIRK21_GAMMA * SDIRK21_GAMMA * SDIRK21_GAMMA
    )
    / (2.0 * (2.0 * SDIRK21_GAMMA - 1.0))
)
SDIRK21_BHAT3 = (
    -2.0
    * SDIRK21_GAMMA
    * SDIRK21_GAMMA
    * (1.0 - SDIRK21_GAMMA + SDIRK21_GAMMA * SDIRK21_GAMMA)
    / (2.0 * SDIRK21_GAMMA - 1.0)
)
SDIRK21_BHAT1 = 1.0 - SDIRK21_BHAT2 - SDIRK21_BHAT3

SDIRK21_TABLEAU = ButcherTableau(
    c=(0.0, 2.0 * SDIRK21_GAMMA, 1.0),
    a=(
        (),
        (SDIRK21_GAMMA, SDIRK21_GAMMA),
        (SDIRK21_B1, SDIRK21_B2, SDIRK21_GAMMA),
    ),
    b=(SDIRK21_B1, SDIRK21_B2, SDIRK21_GAMMA),
    order=2,
    b_embedded=(SDIRK21_BHAT1, SDIRK21_BHAT2, SDIRK21_BHAT3),
    embedded_order=1,
    short_name="SDIRK21",
    full_name="ESDIRK 2(1)",
)

# `SequentialDIRKResolventStep` solves each implicit stage in a diagonal
# stage-increment basis:
#
#     delta_i = known_shift + gamma * dt * f(...)
#
# High, embedded, and error weights are therefore represented as tableau
# weights divided by `gamma`, except for the final diagonal stage contribution,
# which already has unit weight in this increment basis.
_DELTA1_HIGH = SDIRK21_B1 / SDIRK21_GAMMA
_DELTA2_HIGH = SDIRK21_B2 / SDIRK21_GAMMA
_DELTA3_HIGH = 1.0

_DELTA1_LOW = SDIRK21_BHAT1 / SDIRK21_GAMMA
_DELTA2_LOW = SDIRK21_BHAT2 / SDIRK21_GAMMA
_DELTA3_LOW = SDIRK21_BHAT3 / SDIRK21_GAMMA

_STAGE_INCREMENT_WEIGHTS_HIGH = (_DELTA1_HIGH, _DELTA2_HIGH, _DELTA3_HIGH)
_STAGE_INCREMENT_WEIGHTS_LOW = (_DELTA1_LOW, _DELTA2_LOW, _DELTA3_LOW)
_STAGE_INCREMENT_WEIGHTS_ERROR = tuple(
    high - low
    for high, low in zip(
        _STAGE_INCREMENT_WEIGHTS_HIGH,
        _STAGE_INCREMENT_WEIGHTS_LOW,
        strict=True,
    )
)


class SchemeSDIRK21(SchemeBaseImplicitAdaptive):
    """Adaptive ESDIRK 2(1) with sequential implicit stage solves.

    This is a singly diagonally implicit Runge-Kutta pair. The first stage is
    explicit, and each later implicit stage uses the same diagonal coefficient
    `gamma`. That lets STARK solve the stages one at a time through the
    configured resolvent instead of forming one fully coupled nonlinear block.

    The method advances with a second-order formula and estimates local error
    with an embedded first-order formula. It is a compact stiff adaptive
    exemplar because it exercises the implicit solver stack, resolvent failure
    rejection, embedded error control, and adaptive step-size update without the
    coefficient volume of the larger Kvaerno schemes.

    The `_DELTA...` constants are written in the stage-increment basis used by
    `SequentialDIRKResolventStep`, not as raw tableau entries.
    """

    __slots__ = (
        "call_pure",
        "delta1",
        "delta2",
        "delta3",
        "derivative",
        "error",
        "known3",
        "stage1_rate",
        "stepper",
        "trial",
    )

    descriptor = SchemeDescriptor("SDIRK21", "ESDIRK 2(1)")
    tableau = SDIRK21_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        resolvent: Resolvent,
        regulator: Regulator | None = None,
    ) -> None:
        self.derivative = BoundDerivative(derivative)
        self.stepper = SequentialDIRKResolventStep(
            "SDIRK21",
            self.tableau,
            derivative,
            workbench,
            2,
            resolvent,
        )
        self.stage1_rate = workbench.allocate_translation()

        workspace = self.stepper.workspace
        (
            self.delta1,
            self.delta2,
            self.delta3,
            self.trial,
            self.error,
            self.known3,
        ) = workspace.allocate_translation_buffers(6)

        self.initialise_runtime(regulator)
        self.call_pure = self.call_generic
        self.refresh_call()

    @staticmethod
    def default_regulator() -> Regulator:
        return Regulator(error_exponent=0.5)

    def __call__(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        return self.redirect_call(interval, state, executor)

    def call_bind(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        self.assign_executor(executor)
        return self.redirect_call(interval, state, executor)

    def call_monitored(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        accepted_dt = self.call_pure(interval, state, executor)
        report = self.adaptive.report()
        monitor = self.adaptive.monitor

        if monitor is not None:
            monitor(
                MonitorStep(
                    scheme=self.short_name,
                    t_start=report.t_start,
                    t_end=report.t_end,
                    proposed_dt=report.proposed_dt,
                    accepted_dt=report.accepted_dt,
                    next_dt=report.next_dt,
                    error_ratio=report.error_ratio,
                    rejection_count=report.rejection_count,
                )
            )

        return accepted_dt

    def advance_body(self, interval: IntervalLike, state: State) -> None:
        """Compatibility bridge for the transitional adaptive base."""

        self.call_pure(interval, state, Executor())

    def call_generic(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        del executor

        proposal = self.adaptive.propose_step(interval)

        if proposal.remaining <= 0.0:
            self.adaptive.record_stopped(interval)
            return 0.0

        stepper = self.stepper
        workspace = stepper.workspace
        derivative = self.derivative
        scale = workspace.scale
        combine2 = workspace.combine2
        combine3 = workspace.combine3
        apply_delta = workspace.apply_delta
        ratio = self.adaptive.ratio

        assert ratio is not None

        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
        rejection_count = 0

        while True:
            derivative(interval, state, self.stage1_rate)
            delta1 = scale(
                self.delta1,
                dt * SDIRK21_GAMMA,
                self.stage1_rate,
            )

            try:
                delta2 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=0,
                    stage_shift=2.0 * SDIRK21_GAMMA * dt,
                    alpha=dt * SDIRK21_GAMMA,
                    known_shift=delta1,
                    out=self.delta2,
                )

                known3 = combine2(
                    self.known3,
                    _DELTA1_HIGH,
                    delta1,
                    _DELTA2_HIGH,
                    delta2,
                )

                delta3 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=1,
                    stage_shift=dt,
                    alpha=dt * SDIRK21_GAMMA,
                    known_shift=known3,
                    out=self.delta3,
                )

            except ResolventError:
                rejection_count += 1
                dt = self.adaptive.rejected_step(
                    dt,
                    1.0,
                    remaining,
                    self.short_name,
                )
                continue

            delta_high = combine3(
                self.trial,
                _STAGE_INCREMENT_WEIGHTS_HIGH[0],
                delta1,
                _STAGE_INCREMENT_WEIGHTS_HIGH[1],
                delta2,
                _STAGE_INCREMENT_WEIGHTS_HIGH[2],
                delta3,
            )

            error = combine3(
                self.error,
                _STAGE_INCREMENT_WEIGHTS_ERROR[0],
                delta1,
                _STAGE_INCREMENT_WEIGHTS_ERROR[1],
                delta2,
                _STAGE_INCREMENT_WEIGHTS_ERROR[2],
                delta3,
            )

            error_ratio = ratio(error.norm(), delta_high.norm())

            if error_ratio <= 1.0:
                break

            rejection_count += 1
            dt = self.adaptive.rejected_step(
                dt,
                error_ratio,
                remaining,
                self.short_name,
            )

        accepted_dt = dt
        remaining_after = remaining - accepted_dt
        next_dt = self.adaptive.accepted_next_step(
            accepted_dt,
            error_ratio,
            remaining_after,
        )

        interval.step = next_dt
        apply_delta(delta_high, state)

        report = self.adaptive.record_accepted(
            accepted_dt=accepted_dt,
            t_start=t_start,
            proposed_dt=proposed_dt,
            next_dt=next_dt,
            error_ratio=error_ratio,
            rejection_count=rejection_count,
        )
        return report.accepted_dt


__all__ = [
    "SDIRK21_GAMMA",
    "SDIRK21_TABLEAU",
    "SchemeSDIRK21",
]