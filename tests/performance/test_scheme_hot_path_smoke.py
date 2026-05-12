from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import pytest

from stark import Executor, Integrator, Interval, Marcher
from stark.schemes.explicit_fixed.rk4 import SchemeRK4


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


class ScalarWorkbench:
    def allocate_state(self) -> ScalarState:
        return ScalarState()

    def copy_state(self, dst: ScalarState, src: ScalarState) -> None:
        dst.value = src.value

    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


def exponential_growth(interval: Interval, state: ScalarState, out: ScalarTranslation) -> None:
    del interval
    out.value = state.value


def run_rk4_steps(n_steps: int) -> float:
    scheme = SchemeRK4(exponential_growth, ScalarWorkbench())
    marcher = Marcher(scheme, Executor())
    interval = Interval(present=0.0, step=1.0 / n_steps, stop=1.0)
    state = ScalarState(1.0)

    start = perf_counter()
    for _interval, _state in Integrator().live(marcher, interval, state):
        pass
    return perf_counter() - start


@pytest.mark.performance
def test_rk4_hot_path_smoke(benchmark=None) -> None:
    # Run enough steps that dispatch overhead is visible, but keep it cheap.
    n_steps = 2_000

    if benchmark is not None:
        elapsed = benchmark(lambda: run_rk4_steps(n_steps))
    else:
        elapsed = run_rk4_steps(n_steps)

    # This is deliberately loose. It catches catastrophic regressions, not
    # normal machine-to-machine timing variation.
    assert elapsed < 1.0