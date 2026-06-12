from __future__ import annotations

"""Compare two built-in schemes on one small vector-field problem."""

import numpy as np

from stark import (
    Configuration,
    Interval,
    Frame,
    Method,
    System,
)
from stark.diagnostics.comparison import ComparisonEntry, ComparisonProblem, ComparisonRunner
from stark.engines import EngineNumpy
from stark.methods.schemes import SchemeCashKarp, SchemeDormandPrince


def oscillator_rhs(t: float, state, out) -> None:
    del t
    out.dy[0] = state.y[1]
    out.dy[1] = -state.y[0]


system = System(
    derivative=oscillator_rhs,
    frame=Frame({"y": {"translation": "dy", "shape": (2,)}}),
)
template = system.ivp(
    initial={"y": np.array([1.0, 0.0])},
    interval=Interval(present=0.0, step=0.05, stop=1.0),
    method=Method(scheme=SchemeCashKarp),
    engine=EngineNumpy,
    configuration=Configuration(check_progress=False),
)


def diagnostics(state) -> dict[str, float]:
    position, velocity = state.y
    return {"position": float(position), "velocity": float(velocity)}


problem = ComparisonProblem(
    "harmonic oscillator",
    template,
    diagnostics=diagnostics,
)
entries = [
    ComparisonEntry("Cash-Karp", Method(scheme=SchemeCashKarp)),
    ComparisonEntry("Dormand-Prince", Method(scheme=SchemeDormandPrince)),
]

print(ComparisonRunner(problem, entries, repeats=1)())
