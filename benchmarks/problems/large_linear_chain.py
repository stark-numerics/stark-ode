"""Large array-valued linear chain benchmark problem."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from benchmarks.problems.problem import BenchmarkProblemDefinition
from stark import DynamicsStyle, Frame, Interval, System


LARGE_LINEAR_CHAIN_SIZE = 65536


@DynamicsStyle.accepts_instant_writes
def large_linear_chain_rhs(t: float, state, out) -> None:
    out.du[:] = (
        -0.15 * state.u
        + 0.05 * np.roll(state.u, 1)
        - 0.05 * np.roll(state.u, -1)
    )


def large_linear_chain_system() -> System:
    return System(
        dynamics=large_linear_chain_rhs,
        frame=Frame.array("u", translation="du", shape=(LARGE_LINEAR_CHAIN_SIZE,)),
    )


def large_linear_chain_initial() -> Mapping[str, object]:
    x = np.linspace(0.0, 2.0 * np.pi, LARGE_LINEAR_CHAIN_SIZE, endpoint=False)
    return {"u": np.sin(x) + 0.25 * np.sin(3.0 * x)}


def large_linear_chain_interval() -> Interval:
    return Interval(present=0.0, step=1.0e-2, stop=0.1)


BENCHMARK_PROBLEM_LARGE_LINEAR_CHAIN = BenchmarkProblemDefinition(
    name="large-linear-chain",
    summary="Large non-stiff array problem for backend throughput comparisons.",
    system_factory=large_linear_chain_system,
    initial_factory=large_linear_chain_initial,
    interval_factory=large_linear_chain_interval,
)


__all__ = ["BENCHMARK_PROBLEM_LARGE_LINEAR_CHAIN"]

