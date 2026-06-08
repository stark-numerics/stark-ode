from __future__ import annotations

"""Compare schemes and show monitor summaries from the observation pass."""

import numpy as np

from stark import (
    Configuration,
    Interval,
    StarkLayout,
    StarkMethod,
    StarkSystem,
)
from stark.comparison import ComparisonEntry, ComparisonProblem, ComparisonRunner
from stark.engines import StarkEngineNumpy
from stark.schemes import SchemeCashKarp, SchemeDormandPrince


def oscillator_rhs(t: float, state, out) -> None:
    del t
    out.dy[0] = state.y[1]
    out.dy[1] = -state.y[0]


system = StarkSystem(
    derivative=oscillator_rhs,
    layout=StarkLayout({"y": {"translation": "dy", "shape": (2,)}}),
)
template = system.ivp(
    initial={"y": np.array([1.0, 0.0])},
    interval=Interval(present=0.0, step=0.05, stop=1.0),
    method=StarkMethod(scheme=SchemeCashKarp),
    engine=StarkEngineNumpy,
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
        StarkMethod(scheme=SchemeCashKarp),
    ),
    ComparisonEntry(
        "Dormand-Prince",
        StarkMethod(scheme=SchemeDormandPrince),
    ),
]

print(ComparisonRunner(problem, entries, repeats=1)())
