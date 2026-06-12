from __future__ import annotations

# Lesson 2: compare two explicit adaptive schemes
#
# Lesson 1 used `System` to prepare a NumPy Allen-Cahn problem as a
# frame-backed solve. Now we reuse that prepared boundary and ask a
# different question: do two explicit embedded Runge-Kutta schemes agree on the
# final state?
#
# The point of this lesson is validation by comparison. When two different
# adaptive explicit schemes reach nearly the same state, we gain confidence
# that lesson 1 is solving the intended semidiscrete problem rather than merely
# producing a plausible-looking plot.
#
# In a source checkout, run from the `stark-ode` directory with:
#
#     python -m examples.case_studies.allen_cahn.lesson_02_compare_explicit

from stark import Configuration, Method
from stark.comparison import ComparisonRunner, ComparisonEntry, ComparisonProblem
from stark.methods.schemes import SchemeCashKarp, SchemeDormandPrince

from examples.case_studies.allen_cahn.lesson_01_problem import (
    Configuration_TOLERANCE,
    Geometry,
    make_ivp,
    state_diagnostics,
)


if __name__ == "__main__":
    geometry = Geometry()
    configuration = Configuration(scheme_tolerance=Configuration_TOLERANCE)

    # We ask `System` to prepare the problem once, then compare two method
    # recipes against that same IVP boundary. This keeps the comparison focused
    # on method behaviour rather than interface setup.

    template = make_ivp(
        geometry,
        method=Method(scheme=SchemeCashKarp),
        configuration=configuration,
    )

    problem = ComparisonProblem(
        "Allen-Cahn explicit",
        template,
        diagnostics=state_diagnostics,
    )

    # Each entry is a method recipe. The runner asks the IVP to build fresh
    # steppers and fresh states for warmup, timing, and profiling passes.

    entries = [
        ComparisonEntry("Cash-Karp", Method(scheme=SchemeCashKarp)),
        ComparisonEntry("Dormand-Prince", Method(scheme=SchemeDormandPrince)),
    ]

    # The final-state difference table is the main result here. Small
    # off-diagonal differences mean the two schemes agree. The timing/profile
    # sections are secondary at this stage; they become more interesting once
    # implicit and IMEX methods enter the comparison.

    report = ComparisonRunner(problem, entries, repeats=3)()
    print(report)
    print()
    print("What to notice:")
    print("- Agreement between the two explicit schemes is a sanity check on the model.")
    print("- This comparison uses the high-level System-prepared frame path.")
