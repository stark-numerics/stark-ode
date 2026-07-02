"""Compare two built-in methods on the same prepared problem.

`ComparisonRunner` is for short method-selection runs. It prepares each
`ComparisonEntry` from the same `ComparisonProblem`, then reports timing,
final-state differences, diagnostics, monitor summaries, and a coarse profile
breakdown. Use it before a long solve when you want evidence for choosing one
method configuration over another.
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


def diagnostics(state) -> dict[str, float]:
    position, velocity = state.y
    return {"position": float(position), "velocity": float(velocity)}


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
        "harmonic oscillator",
        template,
        diagnostics=diagnostics,
    )


if __name__ == "__main__":
    entries = [
        ComparisonEntry("Cash-Karp", Method(SchemeCashKarp)),
        ComparisonEntry("Dormand-Prince", Method(SchemeDormandPrince)),
    ]
    print(ComparisonRunner(build_problem(), entries, repeats=1)())
