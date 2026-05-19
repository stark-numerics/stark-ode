from __future__ import annotations

"""Compare schemes and show monitor summaries from the observation pass."""

import numpy as np

from stark import Executor, Interval, Marcher
from stark.comparison import Comparator, ComparatorEntry, ComparatorProblem
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


executor = Executor()
problem = ComparatorProblem(
    name="monitored oscillator",
    build_state=build_state,
    build_interval=build_interval,
    difference=difference,
    description=(
        "The comparison runs one monitored observation pass before warmup, "
        "timing, and profiling. Timed repeats stay unmonitored."
    ),
)
entries = [
    ComparatorEntry(
        "Cash-Karp",
        lambda: Marcher(SchemeCashKarp(template.derivative, template.workbench), executor),
    ),
    ComparatorEntry(
        "Dormand-Prince",
        lambda: Marcher(SchemeDormandPrince(template.derivative, template.workbench), executor),
    ),
]

print(Comparator(problem, entries, repeats=1)())
