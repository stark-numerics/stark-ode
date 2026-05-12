from __future__ import annotations

from stark.contracts import Derivative, IntervalLike, Resolvent, State, Workbench
from stark.execution.executor import Executor
from stark.machinery.stage_solve.workers import SequentialDIRKResolventStep
from stark.schemes.base import SchemeBaseImplicitFixed
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.tableau import ButcherTableau


CROUZEIX_DIRK3_GAMMA = 0.5

CROUZEIX_DIRK3_TABLEAU = ButcherTableau(
    c=(0.5, 2.0 / 3.0, 0.5, 1.0),
    a=(
        (0.5,),
        (1.0 / 6.0, 0.5),
        (-0.5, 0.5, 0.5),
        (1.5, -1.5, 0.5, 0.5),
    ),
    b=(1.5, -1.5, 0.5, 0.5),
    order=3,
    short_name="Crouzeix3",
    full_name="Crouzeix DIRK3",
)

# `SequentialDIRKResolventStep` solves each stage in the diagonal
# stage-increment basis:
#
#     delta_i = known_shift + gamma * dt * f(...)
#
# Off-diagonal tableau coefficients therefore enter known shifts as
# `a_ij / gamma`, not as raw `a_ij`.
_DELTA21 = (1.0 / 6.0) / CROUZEIX_DIRK3_GAMMA
_DELTA31 = -0.5 / CROUZEIX_DIRK3_GAMMA
_DELTA32 = 0.5 / CROUZEIX_DIRK3_GAMMA
_DELTA41 = 1.5 / CROUZEIX_DIRK3_GAMMA
_DELTA42 = -1.5 / CROUZEIX_DIRK3_GAMMA
_DELTA43 = 0.5 / CROUZEIX_DIRK3_GAMMA

_STAGE_INCREMENT_WEIGHTS = (_DELTA41, _DELTA42, _DELTA43, 1.0)


class SchemeCrouzeixDIRK3(SchemeBaseImplicitFixed):
    """Crouzeix's fixed-step third-order sequential DIRK method.

    This is a singly diagonally implicit Runge-Kutta scheme: every implicit
    stage uses the same diagonal coefficient, `gamma = 1/2`. STARK represents
    each implicit stage as a shifted one-stage resolvent solve for a stage
    increment `delta_i`.

    The `_DELTA..` constants are therefore not independent method coefficients.
    They are the tableau off-diagonal and final-weight coefficients expressed in
    the stage-increment basis used by `SequentialDIRKResolventStep`; for example,
    `_DELTA21 = a21 / gamma`.

    The call body is intentionally written as four visible stage solves:

    1. solve the first implicit stage from the current state
    2. build the known shift for stage 2 from `delta1`
    3. build the known shift for stage 3 from `delta1` and `delta2`
    4. build the known shift for stage 4 from `delta1`, `delta2`, and `delta3`

    The final update combines the four solved stage increments using the same
    stage-increment representation.
    """

    __slots__ = (
        "call_pure",
        "delta1",
        "delta2",
        "delta3",
        "delta4",
        "known2",
        "known3",
        "known4",
        "redirect_call",
        "stepper",
        "trial",
    )

    descriptor = SchemeDescriptor("Crouzeix3", "Crouzeix DIRK3")
    tableau = CROUZEIX_DIRK3_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        resolvent: Resolvent,
    ) -> None:
        self.stepper = SequentialDIRKResolventStep(
            "Crouzeix DIRK3",
            self.tableau,
            derivative,
            workbench,
            4,
            resolvent,
        )

        workspace = self.stepper.workspace
        (
            self.delta1,
            self.delta2,
            self.delta3,
            self.delta4,
            self.known2,
            self.known3,
            self.known4,
            self.trial,
        ) = workspace.allocate_translation_buffers(8)

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

        stepper = self.stepper
        workspace = stepper.workspace
        scale = workspace.scale
        combine2 = workspace.combine2
        combine3 = workspace.combine3
        combine4 = workspace.combine4

        dt = interval.step if interval.step <= remaining else remaining
        gamma_dt = CROUZEIX_DIRK3_GAMMA * dt

        delta1 = stepper.solve(
            interval,
            state,
            dt,
            block_index=0,
            stage_shift=0.5 * dt,
            alpha=gamma_dt,
            out=self.delta1,
        )

        known2 = scale(self.known2, _DELTA21, delta1)

        delta2 = stepper.solve(
            interval,
            state,
            dt,
            block_index=1,
            stage_shift=(2.0 / 3.0) * dt,
            alpha=gamma_dt,
            known_shift=known2,
            out=self.delta2,
        )

        known3 = combine2(
            self.known3,
            _DELTA31,
            delta1,
            _DELTA32,
            delta2,
        )

        delta3 = stepper.solve(
            interval,
            state,
            dt,
            block_index=2,
            stage_shift=0.5 * dt,
            alpha=gamma_dt,
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
            block_index=3,
            stage_shift=dt,
            alpha=gamma_dt,
            known_shift=known4,
            out=self.delta4,
        )

        delta_high = combine4(
            self.trial,
            _STAGE_INCREMENT_WEIGHTS[0],
            delta1,
            _STAGE_INCREMENT_WEIGHTS[1],
            delta2,
            _STAGE_INCREMENT_WEIGHTS[2],
            delta3,
            _STAGE_INCREMENT_WEIGHTS[3],
            delta4,
        )

        workspace.apply_delta(delta_high, state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)

        return dt


__all__ = [
    "CROUZEIX_DIRK3_GAMMA",
    "CROUZEIX_DIRK3_TABLEAU",
    "SchemeCrouzeixDIRK3",
]