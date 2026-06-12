from __future__ import annotations

"""Compare schemes and show monitor summaries from the observation pass."""

import numpy as np

from stark import (
    Configuration,
    Interval,
    Frame,
    Method,
    System,
)
from stark.comparison import ComparisonEntry, ComparisonProblem, ComparisonRunner
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

problem = ComparisonProblem(
    "monitored oscillator",
    template,
    description=(
        "The comparison runs one monitored observation pass before warmup, "
        "timing, and profiling. Timed repeats stay unmonitored."
    ),
)
entries = [
    ComparisonEntry(
        "Cash-Karp",
        Method(scheme=SchemeCashKarp),
    ),
    ComparisonEntry(
        "Dormand-Prince",
        Method(scheme=SchemeDormandPrince),
    ),
]

print(ComparisonRunner(problem, entries, repeats=1)())
