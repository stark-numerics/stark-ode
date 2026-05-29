from __future__ import annotations

"""Compare two built-in schemes on one small vector problem."""

import numpy as np

from stark import Executor, Interval, Marcher
from stark.comparison import ComparisonRunner, ComparisonEntry, ComparisonProblem
from stark.interface import StarkIVP, StarkVector
from stark.schemes import SchemeCashKarp, SchemeDormandPrince


def oscillator_rhs(t: float, y: np.ndarray) -> np.ndarray:
    del t
    return np.array([y[1], -y[0]])


template = StarkIVP(
    derivative=oscillator_rhs,
    initial=np.array([1.0, 0.0]),
    interval=Interval(present=0.0, step=0.05, stop=1.0),
).build()


def build_state() -> StarkVector:
    return StarkVector(np.array([1.0, 0.0]), template.initial.carrier)


def build_interval() -> Interval:
    return Interval(present=0.0, step=0.05, stop=1.0)


def difference(left: StarkVector, right: StarkVector) -> float:
    return float(np.linalg.norm(left.value - right.value))


def diagnostics(state: StarkVector) -> dict[str, float]:
    position, velocity = state.value
    return {"position": float(position), "velocity": float(velocity)}


executor = Executor()
problem = ComparisonProblem(
    name="harmonic oscillator",
    build_state=build_state,
    build_interval=build_interval,
    difference=difference,
    diagnostics=diagnostics,
)
entries = [
    ComparisonEntry(
        "Cash-Karp",
        Marcher(SchemeCashKarp(template.derivative, template.allocator), executor),
    ),
    ComparisonEntry(
        "Dormand-Prince",
        Marcher(SchemeDormandPrince(template.derivative, template.allocator), executor),
    ),
]

print(ComparisonRunner(problem, entries, repeats=1)())

