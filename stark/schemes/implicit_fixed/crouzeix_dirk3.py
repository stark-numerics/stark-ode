from __future__ import annotations

from stark.schemes.tableau import ButcherTableau
from stark.contracts import Derivative, IntervalLike, Resolvent, State, Workbench
from stark.machinery.stage_solve.workers import SequentialDIRKResolventStep
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.base import SchemeBaseImplicitFixed
from stark.execution.executor import Executor


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

_DELTA21 = (1.0 / 6.0) / CROUZEIX_DIRK3_GAMMA
_DELTA31 = -0.5 / CROUZEIX_DIRK3_GAMMA
_DELTA32 = 0.5 / CROUZEIX_DIRK3_GAMMA
_DELTA41 = 1.5 / CROUZEIX_DIRK3_GAMMA
_DELTA42 = -1.5 / CROUZEIX_DIRK3_GAMMA
_DELTA43 = 0.5 / CROUZEIX_DIRK3_GAMMA
_DELTA_HIGH = (_DELTA41, _DELTA42, _DELTA43, 1.0)


class SchemeCrouzeixDIRK3(SchemeBaseImplicitFixed):
    """
    A fixed-step third-order sequential DIRK scheme attributed to Crouzeix.

    This method uses four sequential implicit stages with a constant diagonal
    coefficient. It is a useful stress case for the current resolvent layer:
    more involved than backward Euler or Crank-Nicolson, but still entirely
    expressible as a sequence of one-stage shifted solves.
    """

    __slots__ = (
        "stepper",
        "delta1",
        "delta2",
        "delta3",
        "delta4",
        "known2",
        "known3",
        "known4",
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
        self.stepper = SequentialDIRKResolventStep("Crouzeix DIRK3", self.tableau, derivative, workbench, 4, resolvent)
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

    def __call__(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        del executor
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        workspace = self.stepper.workspace
        stepper = self.stepper
        scale = workspace.scale
        combine2 = workspace.combine2
        combine3 = workspace.combine3
        combine4 = workspace.combine4
        dt = interval.step if interval.step <= remaining else remaining

        stage1 = stepper.solve(
            interval,
            state,
            dt,
            block_index=0,
            stage_shift=0.5 * dt,
            alpha=CROUZEIX_DIRK3_GAMMA * dt,
            out=self.delta1,
        )
        delta1 = stage1

        known2 = scale(self.known2, _DELTA21, delta1)
        stage2 = stepper.solve(
            interval,
            state,
            dt,
            block_index=1,
            stage_shift=(2.0 / 3.0) * dt,
            alpha=CROUZEIX_DIRK3_GAMMA * dt,
            known_shift=known2,
            out=self.delta2,
        )
        delta2 = stage2

        known3 = combine2(self.known3, _DELTA31, delta1, _DELTA32, delta2)
        stage3 = stepper.solve(
            interval,
            state,
            dt,
            block_index=2,
            stage_shift=0.5 * dt,
            alpha=CROUZEIX_DIRK3_GAMMA * dt,
            known_shift=known3,
            out=self.delta3,
        )
        delta3 = stage3

        known4 = combine3(self.known4, _DELTA41, delta1, _DELTA42, delta2, _DELTA43, delta3)
        stage4 = stepper.solve(
            interval,
            state,
            dt,
            block_index=3,
            stage_shift=dt,
            alpha=CROUZEIX_DIRK3_GAMMA * dt,
            known_shift=known4,
            out=self.delta4,
        )
        delta4 = stage4

        delta_high = combine4(
            self.trial,
            _DELTA_HIGH[0],
            delta1,
            _DELTA_HIGH[1],
            delta2,
            _DELTA_HIGH[2],
            delta3,
            _DELTA_HIGH[3],
            delta4,
        )
        workspace.apply_delta(delta_high, state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)
        return dt


__all__ = ["CROUZEIX_DIRK3_GAMMA", "CROUZEIX_DIRK3_TABLEAU", "SchemeCrouzeixDIRK3"]













