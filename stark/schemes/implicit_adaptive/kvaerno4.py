from __future__ import annotations

from stark.accelerators.binding import BoundDerivative
from stark.contracts import Derivative, IntervalLike, Resolvent, State, Workbench
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator
from stark.machinery.stage_solve.workers import SequentialDIRKResolventStep
from stark.monitor import MonitorStep
from stark.resolvents.failure import ResolventError
from stark.schemes.base import SchemeBaseImplicitAdaptive
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.tableau import ButcherTableau


KVAERNO4_GAMMA = 0.5728160625


def _poly(*coefficients: float) -> float:
    value = 0.0
    for coefficient in coefficients:
        value = value * KVAERNO4_GAMMA + coefficient
    return value


KVAERNO4_A21 = KVAERNO4_GAMMA
KVAERNO4_A31 = (
    _poly(144.0, -180.0, 81.0, -15.0, 1.0)
    * KVAERNO4_GAMMA
    / (_poly(12.0, -6.0, 1.0) ** 2)
)
KVAERNO4_A32 = (
    _poly(-36.0, 39.0, -15.0, 2.0)
    * KVAERNO4_GAMMA
    / (_poly(12.0, -6.0, 1.0) ** 2)
)
KVAERNO4_A41 = _poly(-144.0, 396.0, -330.0, 117.0, -18.0, 1.0) / (
    12.0 * KVAERNO4_GAMMA * KVAERNO4_GAMMA * _poly(12.0, -9.0, 2.0)
)
KVAERNO4_A42 = _poly(72.0, -126.0, 69.0, -15.0, 1.0) / (
    12.0 * KVAERNO4_GAMMA * KVAERNO4_GAMMA * _poly(3.0, -1.0)
)
KVAERNO4_A43 = (
    _poly(-6.0, 6.0, -1.0) * (_poly(12.0, -6.0, 1.0) ** 2)
) / (
    12.0
    * KVAERNO4_GAMMA
    * KVAERNO4_GAMMA
    * _poly(12.0, -9.0, 2.0)
    * _poly(3.0, -1.0)
)
KVAERNO4_A51 = _poly(288.0, -312.0, 120.0, -18.0, 1.0) / (
    48.0 * KVAERNO4_GAMMA * KVAERNO4_GAMMA * _poly(12.0, -9.0, 2.0)
)
KVAERNO4_A52 = _poly(24.0, -12.0, 1.0) / (
    48.0 * KVAERNO4_GAMMA * KVAERNO4_GAMMA * _poly(3.0, -1.0)
)
KVAERNO4_A53 = -(_poly(12.0, -6.0, 1.0) ** 3) / (
    48.0
    * KVAERNO4_GAMMA
    * KVAERNO4_GAMMA
    * _poly(3.0, -1.0)
    * _poly(12.0, -9.0, 2.0)
    * _poly(6.0, -6.0, 1.0)
)
KVAERNO4_A54 = _poly(-24.0, 36.0, -12.0, 1.0) / _poly(
    24.0,
    -24.0,
    4.0,
)

KVAERNO4_C2 = KVAERNO4_GAMMA + KVAERNO4_A21
KVAERNO4_C3 = KVAERNO4_GAMMA + KVAERNO4_A31 + KVAERNO4_A32

KVAERNO4_TABLEAU = ButcherTableau(
    c=(0.0, KVAERNO4_C2, KVAERNO4_C3, 1.0, 1.0),
    a=(
        (),
        (KVAERNO4_A21, KVAERNO4_GAMMA),
        (KVAERNO4_A31, KVAERNO4_A32, KVAERNO4_GAMMA),
        (KVAERNO4_A41, KVAERNO4_A42, KVAERNO4_A43, KVAERNO4_GAMMA),
        (
            KVAERNO4_A51,
            KVAERNO4_A52,
            KVAERNO4_A53,
            KVAERNO4_A54,
            KVAERNO4_GAMMA,
        ),
    ),
    b=(KVAERNO4_A51, KVAERNO4_A52, KVAERNO4_A53, KVAERNO4_A54, KVAERNO4_GAMMA),
    order=4,
    b_embedded=(KVAERNO4_A41, KVAERNO4_A42, KVAERNO4_A43, KVAERNO4_GAMMA, 0.0),
    embedded_order=3,
    short_name="Kvaerno4",
    full_name="Kvaerno 4(3)",
)

# `SequentialDIRKResolventStep` solves each implicit stage in a diagonal
# stage-increment basis:
#
#     delta_i = known_shift + gamma * dt * f(...)
#
# Off-diagonal tableau coefficients, final weights, and embedded weights are
# therefore represented as stage-increment weights by dividing by `gamma`.
_DELTA21 = KVAERNO4_A21 / KVAERNO4_GAMMA
_DELTA31 = KVAERNO4_A31 / KVAERNO4_GAMMA
_DELTA32 = KVAERNO4_A32 / KVAERNO4_GAMMA
_DELTA41 = KVAERNO4_A41 / KVAERNO4_GAMMA
_DELTA42 = KVAERNO4_A42 / KVAERNO4_GAMMA
_DELTA43 = KVAERNO4_A43 / KVAERNO4_GAMMA
_DELTA51 = KVAERNO4_A51 / KVAERNO4_GAMMA
_DELTA52 = KVAERNO4_A52 / KVAERNO4_GAMMA
_DELTA53 = KVAERNO4_A53 / KVAERNO4_GAMMA
_DELTA54 = KVAERNO4_A54 / KVAERNO4_GAMMA

_STAGE_INCREMENT_WEIGHTS_HIGH = (_DELTA51, _DELTA52, _DELTA53, _DELTA54, 1.0)
_STAGE_INCREMENT_WEIGHTS_LOW = (_DELTA41, _DELTA42, _DELTA43, 1.0, 0.0)
_STAGE_INCREMENT_WEIGHTS_ERROR = tuple(
    high - low
    for high, low in zip(
        _STAGE_INCREMENT_WEIGHTS_HIGH,
        _STAGE_INCREMENT_WEIGHTS_LOW,
        strict=True,
    )
)


class SchemeKvaerno4(SchemeBaseImplicitAdaptive):
    """Kvaerno's adaptive ESDIRK 4(3) method.

    This is a fourth-order, stiffly accurate, explicit-first-stage singly
    diagonally implicit Runge-Kutta method with a third-order embedded estimate
    for adaptive step control.

    The first stage is explicit. The remaining stages share the same diagonal
    coefficient `gamma`, so STARK solves them sequentially through
    `SequentialDIRKResolventStep` rather than through one coupled nonlinear
    system.

    The coefficient definitions are deliberately left in formula form. They are
    generated from Kvaerno's construction and are easier to audit this way than
    as unexplained decimal constants.

    The `_DELTA...` constants are written in the stage-increment basis used by
    the stepper. They are derived from the tableau coefficients, not separate
    method parameters.

    Further reading: Anne Kvaerno, "Singly diagonally implicit Runge-Kutta
    methods with an explicit first stage" (BIT Numerical Mathematics, 2004).
    """

    __slots__ = (
        "call_pure",
        "delta1",
        "delta2",
        "delta3",
        "delta4",
        "delta5",
        "derivative",
        "error",
        "known4",
        "known5",
        "stage1_rate",
        "stepper",
        "trial",
    )

    descriptor = SchemeDescriptor("Kvaerno4", "Kvaerno 4(3)")
    tableau = KVAERNO4_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        resolvent: Resolvent,
        regulator: Regulator | None = None,
    ) -> None:
        self.derivative = BoundDerivative(derivative)
        self.stepper = SequentialDIRKResolventStep(
            "Kvaerno4",
            self.tableau,
            derivative,
            workbench,
            4,
            resolvent,
        )
        self.stage1_rate = workbench.allocate_translation()

        workspace = self.stepper.workspace
        (
            self.delta1,
            self.delta2,
            self.delta3,
            self.delta4,
            self.delta5,
            self.known4,
            self.known5,
            self.trial,
            self.error,
        ) = workspace.allocate_translation_buffers(9)

        self.initialise_runtime(regulator)
        self.call_pure = self.call_generic
        self.refresh_call()

    @staticmethod
    def default_regulator() -> Regulator:
        return Regulator(error_exponent=0.25)

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
        combine4 = workspace.combine4
        combine5 = workspace.combine5
        apply_delta = workspace.apply_delta
        ratio = self.adaptive.ratio

        assert ratio is not None

        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
        rejection_count = 0

        derivative(interval, state, self.stage1_rate)

        while True:
            delta1 = scale(
                self.delta1,
                dt * KVAERNO4_GAMMA,
                self.stage1_rate,
            )

            try:
                delta2 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=0,
                    stage_shift=KVAERNO4_C2 * dt,
                    alpha=dt * KVAERNO4_GAMMA,
                    known_shift=delta1,
                    out=self.delta2,
                )

                known3 = combine2(
                    self.trial,
                    _DELTA31,
                    delta1,
                    _DELTA32,
                    delta2,
                )

                delta3 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=1,
                    stage_shift=KVAERNO4_C3 * dt,
                    alpha=dt * KVAERNO4_GAMMA,
                    known_shift=known3,
                    out=self.delta3,
                )

                known4 = combine3(
                    self.known4,
                    _DELTA41,
                    delta1,
                    _DELTA42,
                    delta2,
                    _DELTA43,
                    delta3,
                )

                delta4 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=2,
                    stage_shift=dt,
                    alpha=dt * KVAERNO4_GAMMA,
                    known_shift=known4,
                    out=self.delta4,
                )

                known5 = combine4(
                    self.known5,
                    _DELTA51,
                    delta1,
                    _DELTA52,
                    delta2,
                    _DELTA53,
                    delta3,
                    _DELTA54,
                    delta4,
                )

                delta5 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=3,
                    stage_shift=dt,
                    alpha=dt * KVAERNO4_GAMMA,
                    known_shift=known5,
                    out=self.delta5,
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

            delta_high = combine5(
                self.trial,
                _STAGE_INCREMENT_WEIGHTS_HIGH[0],
                delta1,
                _STAGE_INCREMENT_WEIGHTS_HIGH[1],
                delta2,
                _STAGE_INCREMENT_WEIGHTS_HIGH[2],
                delta3,
                _STAGE_INCREMENT_WEIGHTS_HIGH[3],
                delta4,
                _STAGE_INCREMENT_WEIGHTS_HIGH[4],
                delta5,
            )

            error = combine5(
                self.error,
                _STAGE_INCREMENT_WEIGHTS_ERROR[0],
                delta1,
                _STAGE_INCREMENT_WEIGHTS_ERROR[1],
                delta2,
                _STAGE_INCREMENT_WEIGHTS_ERROR[2],
                delta3,
                _STAGE_INCREMENT_WEIGHTS_ERROR[3],
                delta4,
                _STAGE_INCREMENT_WEIGHTS_ERROR[4],
                delta5,
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
    "KVAERNO4_GAMMA",
    "KVAERNO4_TABLEAU",
    "SchemeKvaerno4",
]