"""Coupled non-stiff oscillator benchmark problem."""

from __future__ import annotations

from collections.abc import Mapping
from math import cos, sin

import numpy as np

from benchmarks.problems.problem import BenchmarkProblemDefinition
from stark import DynamicsStyle, Frame, Interval, System


@DynamicsStyle.accepts_instant_writes
def harmonic_oscillator_rhs(t: float, state, out) -> None:
    out.dx[:] = state.v
    out.dv[:] = -state.x


def harmonic_oscillator_system() -> System:
    return System(
        dynamics=harmonic_oscillator_rhs,
        frame=Frame.from_fields(
            ("x", "dx", (1,)),
            ("v", "dv", (1,)),
        ),
    )


def harmonic_oscillator_initial() -> Mapping[str, object]:
    return {"x": np.array([1.0]), "v": np.array([0.0])}


def harmonic_oscillator_interval() -> Interval:
    return Interval(present=0.0, step=0.05, stop=1.0)


def harmonic_oscillator_reference() -> Mapping[str, object]:
    return {"x": np.array([cos(1.0)]), "v": np.array([-sin(1.0)])}


def harmonic_oscillator_error(state: object, reference: Mapping[str, object]) -> float:
    x_target = reference["x"]
    v_target = reference["v"]
    return float(max(abs(state.x[0] - x_target[0]), abs(state.v[0] - v_target[0])))


BENCHMARK_PROBLEM_HARMONIC_OSCILLATOR = BenchmarkProblemDefinition(
    name="harmonic-oscillator",
    summary="Non-stiff coupled oscillator with an exact trigonometric reference.",
    system_factory=harmonic_oscillator_system,
    initial_factory=harmonic_oscillator_initial,
    interval_factory=harmonic_oscillator_interval,
    reference_factory=harmonic_oscillator_reference,
    final_error=harmonic_oscillator_error,
)


__all__ = ["BENCHMARK_PROBLEM_HARMONIC_OSCILLATOR"]

