"""Small stiff Robertson kinetics benchmark problem."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

import numpy as np

from benchmarks.problems.problem import BenchmarkProblemDefinition
from stark import DerivativeStyle, Frame, Interval, LinearizerStyle, System


Array: TypeAlias = np.ndarray


@DerivativeStyle.kernel_accepts_instant_writes(state=("y",), translation=("dy",))
def robertson_rhs(t: float, y: Array, dy: Array) -> None:
    y1 = y[0]
    y2 = y[1]
    y3 = y[2]
    coupling = 1.0e4 * y2 * y3
    quadratic = 3.0e7 * y2 * y2
    dy[0] = -0.04 * y1 + coupling
    dy[1] = 0.04 * y1 - coupling - quadratic
    dy[2] = quadratic


@LinearizerStyle.kernel_accepts_instant_writes(
    state=("y",),
    source=("dy",),
    target=("dy",),
)
def robertson_jacobian(t: float, y: Array, source: Array, target: Array) -> None:
    y2 = y[1]
    y3 = y[2]
    coupling = 1.0e4 * (source[1] * y3 + y2 * source[2])
    quadratic = 6.0e7 * y2 * source[1]
    target[0] = -0.04 * source[0] + coupling
    target[1] = 0.04 * source[0] - coupling - quadratic
    target[2] = quadratic


def robertson_system() -> System:
    return System(
        derivative=robertson_rhs,
        frame=Frame.array("y", translation="dy", shape=(3,)),
        linearizer=robertson_jacobian,
    )


def robertson_initial() -> Mapping[str, object]:
    return {"y": np.array([1.0, 0.0, 0.0])}


def robertson_interval() -> Interval:
    return Interval(present=0.0, step=1.0e-3, stop=1.0)


BENCHMARK_PROBLEM_ROBERTSON = BenchmarkProblemDefinition(
    name="robertson",
    summary="Small stiff chemical kinetics problem.",
    system_factory=robertson_system,
    initial_factory=robertson_initial,
    interval_factory=robertson_interval,
)


__all__ = ["BENCHMARK_PROBLEM_ROBERTSON"]

