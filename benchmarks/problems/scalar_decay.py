"""Scalar exponential decay benchmark problem."""

from __future__ import annotations

from math import exp
from collections.abc import Mapping

import numpy as np

from benchmarks.problems.problem import BenchmarkProblemDefinition
from stark import Frame, Interval, System


def scalar_decay_rhs(t: float, state, out) -> None:
    out.dy[0] = -0.5 * state.y[0]


def scalar_decay_system() -> System:
    return System(
        derivative=scalar_decay_rhs,
        frame=Frame.scalar("y", translation="dy"),
    )


def scalar_decay_initial() -> Mapping[str, object]:
    return {"y": np.array([2.0])}


def scalar_decay_interval() -> Interval:
    return Interval(present=0.0, step=0.1, stop=1.0)


def scalar_decay_reference() -> Mapping[str, object]:
    return {"y": np.array([2.0 * exp(-0.5)])}


def scalar_decay_error(state: object, reference: Mapping[str, object]) -> float:
    target = reference["y"]
    return float(abs(state.y[0] - target[0]))


BENCHMARK_PROBLEM_SCALAR_DECAY = BenchmarkProblemDefinition(
    name="scalar-decay",
    summary="Tiny non-stiff scalar IVP with an exact exponential reference.",
    system_factory=scalar_decay_system,
    initial_factory=scalar_decay_initial,
    interval_factory=scalar_decay_interval,
    reference_factory=scalar_decay_reference,
    final_error=scalar_decay_error,
)


__all__ = ["BENCHMARK_PROBLEM_SCALAR_DECAY"]

