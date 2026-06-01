from __future__ import annotations

from dataclasses import dataclass

from stark import Interval
from stark.block import Block
from stark.resolvents import ResolventAnderson, ResolventPolicy, ResolventTolerance
from stark.schemes.requests.resolvent import SchemeResolventRequest


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

    def __add__(self, other: ScalarTranslation) -> ScalarTranslation:
        return ScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> ScalarTranslation:
        return ScalarTranslation(scalar * self.value)

    @staticmethod
    def scale(a: float, x: ScalarTranslation, out: ScalarTranslation) -> ScalarTranslation:
        out.value = a * x.value
        return out

    @staticmethod
    def combine2(
        a0: float,
        x0: ScalarTranslation,
        a1: float,
        x1: ScalarTranslation,
        out: ScalarTranslation,
    ) -> ScalarTranslation:
        out.value = a0 * x0.value + a1 * x1.value
        return out

    @staticmethod
    def combine3(
        a0: float,
        x0: ScalarTranslation,
        a1: float,
        x1: ScalarTranslation,
        a2: float,
        x2: ScalarTranslation,
        out: ScalarTranslation,
    ) -> ScalarTranslation:
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


def inner_product(left: ScalarTranslation, right: ScalarTranslation) -> float:
    return left.value * right.value


def test_anderson_resolvent_solves_rhs_shift_without_reversing_residual_arguments() -> None:
    resolvent = ResolventAnderson(
        ScalarAllocator(),
        inner_product,
        ExecutorTolerance=ResolventTolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=4),
    )
    interval = Interval(present=0.0, step=0.1, stop=1.0)
    state = ScalarState(1.0)
    rhs = Block([ScalarTranslation(2.0)])
    out = Block([ScalarTranslation()])
    problem = SchemeResolventRequest(zero_rhs, interval, state, rhs, 0.0)

    resolvent(problem, out)

    assert out[0].value == 2.0
