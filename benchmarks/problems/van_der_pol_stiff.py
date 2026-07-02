"""Stiff Van der Pol oscillator benchmark problem."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from benchmarks.problems.problem import BenchmarkProblemDefinition
from stark import DerivativeStyle, Frame, Interval, LinearizerStyle, System


VAN_DER_POL_MU = 25.0


@DerivativeStyle.accepts_instant_writes
def van_der_pol_rhs(t: float, state, out) -> None:
    out.dx[:] = state.v
    out.dv[:] = VAN_DER_POL_MU * (1.0 - state.x * state.x) * state.v - state.x


@LinearizerStyle.kernel_accepts_instant_writes(
    state=("x", "v"),
    source=("dx", "dv"),
    target=("dx", "dv"),
)
def van_der_pol_jacobian(
    t: float,
    x,
    v,
    source_dx,
    source_dv,
    target_dx,
    target_dv,
) -> None:
    target_dx[:] = source_dv
    target_dv[:] = (
        (-2.0 * VAN_DER_POL_MU * x * v - 1.0) * source_dx
        + VAN_DER_POL_MU * (1.0 - x * x) * source_dv
    )


def van_der_pol_system() -> System:
    return System(
        derivative=van_der_pol_rhs,
        frame=Frame.from_fields(
            ("x", "dx", (1,)),
            ("v", "dv", (1,)),
        ),
        linearizer=van_der_pol_jacobian,
    )


def van_der_pol_initial() -> Mapping[str, object]:
    return {"x": np.array([2.0]), "v": np.array([0.0])}


def van_der_pol_interval() -> Interval:
    return Interval(present=0.0, step=1.0e-3, stop=0.5)


BENCHMARK_PROBLEM_VAN_DER_POL_STIFF = BenchmarkProblemDefinition(
    name="van-der-pol-stiff",
    summary="Stiff coupled oscillator for implicit and adaptive-step behaviour.",
    system_factory=van_der_pol_system,
    initial_factory=van_der_pol_initial,
    interval_factory=van_der_pol_interval,
)


__all__ = ["BENCHMARK_PROBLEM_VAN_DER_POL_STIFF"]

