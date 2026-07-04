from __future__ import annotations

from dataclasses import dataclass

from stark.core import Interval
from stark.core.block import Block
from stark.core.contracts import IntervalLike
from stark.methods.schemes.request import SchemeResolventRequest


class DummyTranslation:
    def __call__(self, origin: DummyState, result: DummyState) -> None:
        result.value = origin.value + self.value

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def norm(self) -> float:
        return 0.0

    def __add__(self, other: "DummyTranslation") -> "DummyTranslation":
        return DummyTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> "DummyTranslation":
        return DummyTranslation(scalar * self.value)


@dataclass(slots=True)
class DummyState:
    value: float = 0.0


def dynamics(interval: IntervalLike, state: DummyState, out: DummyTranslation) -> None:
    del interval, state, out


def test_scheme_stage_problem_preserves_scheme_constructed_data() -> None:
    interval = Interval(present=0.0, step=0.1, stop=1.0)
    origin = DummyState()
    rhs = Block([DummyTranslation()])

    problem = SchemeResolventRequest(
        dynamics=dynamics,
        interval=interval,
        origin=origin,
        rhs=rhs,
        alpha=0.5,
    )

    assert problem.dynamics is dynamics
    assert problem.interval is interval
    assert problem.origin is origin
    assert problem.rhs is rhs
    assert problem.alpha == 0.5
