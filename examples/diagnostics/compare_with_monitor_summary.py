"""Compare methods and include monitor summaries from an observation pass.

Comparison separates observation from timing. The monitored pass explains what
the schemes did, while warmup, timed repeats, and profiling are run separately
so measurement conditions remain visible. This is the example to read when a
timing table alone is too opaque.
"""

from __future__ import annotations

import numpy as np

from stark import Frame, Interval, Method, System
from stark.diagnostics.comparison import ComparisonEntry, ComparisonProblem, ComparisonRunner
from stark.engines import EngineNumpy
from stark.methods import SchemeCashKarp, SchemeDormandPrince


def oscillator_rhs(t: float, state, out) -> None:
    del t
    out.dy[0] = state.y[1]
    out.dy[1] = -state.y[0]


def build_problem() -> ComparisonProblem:
    system = System(
        derivative=oscillator_rhs,
        frame=Frame.vector("y", translation="dy", length=2),
    )
    template = system.ivp(
        initial={"y": np.array([1.0, 0.0])},
        interval=Interval(present=0.0, step=0.05, stop=1.0),
        method=Method(SchemeCashKarp),
        engine=EngineNumpy,
    )
    return ComparisonProblem(
        "monitored oscillator",
        template,
        description=(
            "The comparison runs one monitored observation pass before warmup, "
            "timing, and profiling. Timed repeats stay unmonitored."
        ),
    )


if __name__ == "__main__":
    entries = [
        ComparisonEntry("Cash-Karp", Method(SchemeCashKarp)),
        ComparisonEntry("Dormand-Prince", Method(SchemeDormandPrince)),
    ]
    print(ComparisonRunner(build_problem(), entries, repeats=1)())
