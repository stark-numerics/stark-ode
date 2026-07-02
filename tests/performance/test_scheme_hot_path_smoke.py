from __future__ import annotations

from time import perf_counter

import pytest

from stark import Interval
from stark.core import Integrator, IntegratorStepper
from stark.methods.schemes.explicit.fixed.rk4 import SchemeRK4
from tests.support import (
    DummyScalarAllocator,
    DummyScalarState,
    dummy_exponential_growth_rhs,
)


def run_rk4_steps(n_steps: int) -> float:
    scheme = SchemeRK4(dummy_exponential_growth_rhs, DummyScalarAllocator())
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=1.0 / n_steps, stop=1.0)
    state = DummyScalarState(1.0)

    start = perf_counter()
    for _interval, _state in Integrator().mutating_trajectory(stepper, interval, state):
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
