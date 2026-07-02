from __future__ import annotations

from dataclasses import dataclass
from stark.core.contracts import IntervalLike
from stark.methods.resolvents.requests.resolvent import ResolventRequest
from stark.methods.schemes.request import SchemeResolventRequest


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

    def copy(self) -> "DummyInterval":
        return DummyInterval(self.present, self.step, self.stop)

    def increment(self, dt: float) -> None:
        del dt
        raise TypeError("Frozen interval fixture cannot be advanced.")


def derivative(interval: IntervalLike, state: object, out: DummyTranslation) -> None:
    del interval, state, out


def accepts_stage_problem(
    problem: ResolventRequest,
) -> ResolventRequest:
    return problem


def test_scheme_stage_problem_satisfies_resolvent_stage_problem() -> None:
    interval = DummyInterval(0.0, 0.1, 1.0)
    origin = object()
    rhs = DummyTranslation()

    problem = SchemeResolventRequest(
        derivative=derivative,
        interval=interval,
        origin=origin,
        rhs=rhs,
        alpha=0.25,
    )

    assert accepts_stage_problem(problem) is problem
    assert problem.derivative is derivative
    assert problem.interval is interval
    assert problem.origin is origin
    assert problem.rhs is rhs
    assert problem.alpha == 0.25
