from __future__ import annotations

# Lesson 2: compare two explicit adaptive schemes
#
# Lesson 1 used `StarkIVP` to prepare a NumPy Allen-Cahn problem as a
# `StarkVector` solve. Now we reuse that prepared vector boundary and ask a
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

from stark import Executor, Marcher
from stark.comparison import Comparator, ComparatorEntry, ComparatorProblem
from stark.interface import StarkDerivative, StarkIVP, StarkVector
from stark.schemes import SchemeCashKarp, SchemeDormandPrince

from examples.case_studies.allen_cahn.lesson_01_problem import (
    DIFFUSIVITY,
    TOLERANCE,
    AllenCahnRHS,
    Geometry,
    initial_profile,
    make_interval,
    state_diagnostics,
    state_difference,
)


if __name__ == "__main__":
    geometry = Geometry()
    executor = Executor(tolerance=TOLERANCE)

    # We ask `StarkIVP` to prepare the vector-space boundary once, then reuse
    # the prepared derivative/workbench with two different schemes. This keeps
    # the comparison focused on method behaviour rather than interface setup.

    template = StarkIVP(
        derivative=StarkDerivative.in_place(
            AllenCahnRHS(geometry, DIFFUSIVITY),
        ),
        initial=initial_profile(geometry),
        interval=make_interval(),
        scheme=SchemeCashKarp,
        executor=executor,
    ).build()

    carrier = template.initial.carrier
    derivative = template.derivative
    workbench = template.workbench

    # Comparator needs fresh states and intervals for each warmup, timed repeat,
    # and profiling pass. The carrier prepared by `StarkIVP` knows how to treat
    # these NumPy arrays as STARK vectors.

    problem = ComparatorProblem(
        name="Allen-Cahn explicit",
        build_state=lambda: StarkVector(initial_profile(geometry), carrier),
        build_interval=make_interval,
        difference=state_difference,
        diagnostics=state_diagnostics,
    )

    # Both entries share the same prepared derivative and workbench. That is
    # acceptable here because the derivative owns only reusable scratch storage
    # for one sequential comparison run.

    entries = [
        ComparatorEntry(
            "Cash-Karp",
            Marcher(SchemeCashKarp(derivative, workbench), executor),
        ),
        ComparatorEntry(
            "Dormand-Prince",
            Marcher(SchemeDormandPrince(derivative, workbench), executor),
        ),
    ]

    # The final-state difference table is the main result here. Small
    # off-diagonal differences mean the two schemes agree. The timing/profile
    # sections are secondary at this stage; they become more interesting once
    # implicit and IMEX methods enter the comparison.

    report = Comparator(problem, entries, repeats=3)()
    print(report)
    print()
    print("What to notice:")
    print("- Agreement between the two explicit schemes is a sanity check on the model.")
    print("- This comparison still uses the high-level StarkIVP-prepared vector path.")
