"""Array-valued split reaction-diffusion benchmark problem."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from benchmarks.problems.problem import BenchmarkProblemDefinition
from stark import Derivative, DerivativeStyle, Frame, Interval, LinearizerStyle, System


REACTION_DIFFUSION_GRID_SIZE = 64
REACTION_DIFFUSION_LENGTH = 6.0
REACTION_DIFFUSION_DIFFUSIVITY = 0.08


def reaction_diffusion_initial_profile() -> np.ndarray:
    x = np.linspace(0.0, REACTION_DIFFUSION_LENGTH, REACTION_DIFFUSION_GRID_SIZE, endpoint=False)
    wave_number = 2.0 * np.pi / REACTION_DIFFUSION_LENGTH
    return 0.5 * np.sin(wave_number * x) + 0.5 * np.sin(3.0 * wave_number * x)


def reaction_diffusion_laplacian(u: np.ndarray) -> np.ndarray:
    dx = REACTION_DIFFUSION_LENGTH / REACTION_DIFFUSION_GRID_SIZE
    return (np.roll(u, 1) - 2.0 * u + np.roll(u, -1)) / (dx * dx)


@DerivativeStyle.accepts_instant_writes
def reaction_diffusion_implicit_rhs(t: float, state, out) -> None:
    dx = REACTION_DIFFUSION_LENGTH / REACTION_DIFFUSION_GRID_SIZE
    out.du[:] = REACTION_DIFFUSION_DIFFUSIVITY * (
        np.roll(state.u, 1) - 2.0 * state.u + np.roll(state.u, -1)
    ) / (dx * dx)


@DerivativeStyle.accepts_instant_writes
def reaction_diffusion_explicit_rhs(t: float, state, out) -> None:
    out.du[:] = state.u - state.u * state.u * state.u


@LinearizerStyle.kernel_accepts_instant_writes(
    state=("u",),
    source=("du",),
    target=("du",),
)
def reaction_diffusion_implicit_jacobian(t: float, u, source, target) -> None:
    dx = REACTION_DIFFUSION_LENGTH / REACTION_DIFFUSION_GRID_SIZE
    target[:] = REACTION_DIFFUSION_DIFFUSIVITY * (
        np.roll(source, 1) - 2.0 * source + np.roll(source, -1)
    ) / (dx * dx)


def reaction_diffusion_system() -> System:
    return System(
        derivative=Derivative.split(
            implicit=Derivative(reaction_diffusion_implicit_rhs),
            explicit=Derivative(reaction_diffusion_explicit_rhs),
        ),
        frame=Frame.array("u", translation="du", shape=(REACTION_DIFFUSION_GRID_SIZE,)),
        linearizer=reaction_diffusion_implicit_jacobian,
    )


def reaction_diffusion_initial() -> Mapping[str, object]:
    return {"u": reaction_diffusion_initial_profile()}


def reaction_diffusion_interval() -> Interval:
    return Interval(present=0.0, step=1.0e-3, stop=0.1)


BENCHMARK_PROBLEM_REACTION_DIFFUSION_ARRAY = BenchmarkProblemDefinition(
    name="reaction-diffusion-array",
    summary="Array-valued split problem for IMEX schemes and array-backed engines.",
    system_factory=reaction_diffusion_system,
    initial_factory=reaction_diffusion_initial,
    interval_factory=reaction_diffusion_interval,
)


__all__ = ["BENCHMARK_PROBLEM_REACTION_DIFFUSION_ARRAY"]

