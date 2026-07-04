from __future__ import annotations

from dataclasses import dataclass

from stark.core import Interval
from stark.core.block import Block
from stark.core.contracts import IntervalLike
from stark.methods.resolvents.requests.resolvent import ResolventRequest
from stark.methods.schemes.request import SchemeResolventRequest


@dataclass(slots=True)
class RequestState:
    """State fixture for resolvent request contract tests."""

    value: float = 0.0


@dataclass(slots=True)
class RequestTranslation:
    """Translation fixture used inside block-valued resolvent requests."""

    value: float = 0.0

    def __call__(self, origin: RequestState, result: RequestState) -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: RequestTranslation) -> RequestTranslation:
        return RequestTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> RequestTranslation:
        return RequestTranslation(scalar * self.value)


def derivative(interval: IntervalLike, state: RequestState, out: RequestTranslation) -> None:
    del interval, state, out


def accepts_stage_problem(
    problem: ResolventRequest[RequestState, RequestTranslation],
) -> ResolventRequest[RequestState, RequestTranslation]:
    return problem


def test_scheme_stage_problem_satisfies_resolvent_stage_problem() -> None:
    interval = Interval(present=0.0, step=0.1, stop=1.0)
    origin = RequestState()
    rhs = Block[RequestTranslation]([RequestTranslation()])

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
