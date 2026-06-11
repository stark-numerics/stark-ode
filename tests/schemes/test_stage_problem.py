from __future__ import annotations

from dataclasses import dataclass

from stark.contracts import IntervalLike
from stark.methods.schemes.requests.resolvent import SchemeResolventRequest


class DummyTranslation:
    def __call__(self, origin: object, result: object) -> None:
        del origin, result

    def norm(self) -> float:
        return 0.0

    def __add__(self, other: "DummyTranslation") -> "DummyTranslation":
        del other
        return DummyTranslation()

    def __rmul__(self, scalar: float) -> "DummyTranslation":
        del scalar
        return DummyTranslation()


@dataclass(frozen=True, slots=True)
class DummyInterval:
    present: float
    step: float
    stop: float


def derivative(interval: IntervalLike, state: object, out: DummyTranslation) -> DummyTranslation:
    del interval, state
    return out


def test_scheme_stage_problem_preserves_scheme_constructed_data() -> None:
    interval = DummyInterval(0.0, 0.1, 1.0)
    origin = object()
    rhs = DummyTranslation()

    problem = SchemeResolventRequest(
        derivative=derivative,
        interval=interval,
        origin=origin,
        rhs=rhs,
        alpha=0.5,
    )

    assert problem.derivative is derivative
    assert problem.interval is interval
    assert problem.origin is origin
    assert problem.rhs is rhs
    assert problem.alpha == 0.5
