from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark import Interval, Monitor
from stark.block import Block
from stark.methods.resolvents import ResolventPicard
from stark.methods.schemes.requests.resolvent import SchemeResolventRequest


@dataclass(slots=True)
class ScalarState:
    value: float = 0.0


@dataclass(slots=True)
class ScalarTranslation:
    value: float = 0.0

    def __call__(self, origin: ScalarState, result: ScalarState) -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: "ScalarTranslation") -> "ScalarTranslation":
        return ScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> "ScalarTranslation":
        return ScalarTranslation(scalar * self.value)

    @staticmethod
    def scale(a: float, x: "ScalarTranslation", out: "ScalarTranslation") -> "ScalarTranslation":
        out.value = a * x.value
        return out

    @staticmethod
    def combine2(
        a0: float,
        x0: "ScalarTranslation",
        a1: float,
        x1: "ScalarTranslation",
        out: "ScalarTranslation",
    ) -> "ScalarTranslation":
        out.value = a0 * x0.value + a1 * x1.value
        return out

    @staticmethod
    def combine3(
        a0: float,
        x0: "ScalarTranslation",
        a1: float,
        x1: "ScalarTranslation",
        a2: float,
        x2: "ScalarTranslation",
        out: "ScalarTranslation",
    ) -> "ScalarTranslation":
        out.value = a0 * x0.value + a1 * x1.value + a2 * x2.value
        return out

    linear_combine = [scale, combine2, combine3]


class ScalarAllocator:
    def allocate_state(self) -> ScalarState:
        return ScalarState()

    def copy_state(self, source: ScalarState, out: ScalarState) -> None:
        out.value = source.value

    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


def zero_rhs(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 0.0


def test_picard_records_resolvent_solve_when_monitor_assigned() -> None:
    monitor = Monitor()
    resolvent = ResolventPicard(ScalarAllocator())
    interval = Interval(present=0.0, step=0.1, stop=1.0)
    state = ScalarState()
    out = Block([ScalarTranslation()])
    problem = SchemeResolventRequest(zero_rhs, interval, state, None, 0.1)

    resolvent.assign_monitor(monitor.resolvent)
    resolvent(problem, out)

    assert len(monitor.resolvent.solves) == 1

    solve = monitor.resolvent.solves[0]
    assert solve.resolvent == "Picard"
    assert solve.alpha == pytest.approx(0.1)
    assert solve.block_size == 1
    assert solve.iteration_count == 0
    assert solve.error == pytest.approx(0.0)
    assert solve.scale == pytest.approx(0.0)
    assert solve.converged is True


def test_picard_unassign_monitor_restores_unmonitored_solves() -> None:
    monitor = Monitor()
    resolvent = ResolventPicard(ScalarAllocator())
    interval = Interval(present=0.0, step=0.1, stop=1.0)
    state = ScalarState()
    out = Block([ScalarTranslation()])
    problem = SchemeResolventRequest(zero_rhs, interval, state, None, 0.1)

    resolvent.assign_monitor(monitor.resolvent)
    resolvent.unassign_monitor()
    resolvent(problem, out)

    assert monitor.resolvent.solves == []
