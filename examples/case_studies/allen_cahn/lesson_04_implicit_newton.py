from __future__ import annotations

# Lesson 4: fully implicit Newton solve
#
# Explicit adaptive methods are easy to set up through `StarkIVP`, but diffusion
# can make them conservative. This lesson drops down to the prepared
# `StarkVector` boundary and treats the full Allen-Cahn right-hand side
# implicitly with SDIRK21 and a Newton resolvent.
#
# A Newton resolvent needs two problem-specific pieces:
#
# - a linearizer that exposes the local Jacobian action
# - an inverter that solves the resulting linear systems
#
# The point is to test the obvious stiff-PDE instinct: make diffusion easier by
# using an implicit method. The comparison is intentionally not guaranteed to
# flatter the implicit solve. Nonlinear Newton work and Krylov linear solves can
# cost more than the explicit method saves on accepted step count.
#
# In a source checkout, run from the `stark-ode` directory with:
#
#     python -m examples.case_studies.allen_cahn.lesson_04_implicit_newton

import numpy as np

from stark import Executor, Interval, Marcher
from stark.comparison import ComparisonRunner, ComparisonEntry, ComparisonProblem
from stark.interface import StarkDerivative, StarkIVP, StarkVector
from stark.interface.vector import StarkVectorTranslation
from stark.inverters import InverterBiCGStab, InverterPolicy, InverterTolerance
from stark.resolvents import ResolventNewton, ResolventPolicy, ResolventTolerance
from stark.schemes import SchemeCashKarp, SchemeSDIRK21

from examples.case_studies.allen_cahn.lesson_01_problem import (
    ACCELERATOR,
    DIFFUSIVITY,
    EXECUTOR_TOLERANCE,
    AllenCahnRHS,
    Geometry,
    initial_profile,
    make_interval,
    state_diagnostics,
    state_difference,
)


# A Newton method needs a linear approximation to the derivative. STARK asks
# for that as an operator action, not as a dense matrix. The operator below
# represents
#
#     J(u) v = D L v + (1 - 3 u^2) v
#
# where L is the same periodic finite-difference Laplacian from lesson 1.
#
# This comes from expanding the full grid derivative
#
#     f(u) = D L u + u - u^3
#
# around the current state:
#
#     f(u + delta u)
#       = f(u) + D L delta u + (1 - 3 u^2) delta u + O(delta u^2).
#
# The linearizer is the part multiplying `delta u`. STARK does not need a dense
# Jacobian matrix; it only needs a callable that writes J(u)[v] into an output
# translation.


class AllenCahnJacobianOperator:
    __slots__ = ("diffusivity", "inv_dx2", "laplacian_u", "u")

    def __init__(self, geometry: Geometry, diffusivity: float) -> None:
        self.diffusivity = diffusivity
        self.inv_dx2 = 1.0 / (geometry.dx * geometry.dx)
        self.laplacian_u = np.zeros(geometry.grid_size, dtype=np.float64)
        self.u: np.ndarray | None = None

    def configure(self, state: StarkVector) -> None:
        # The resolvent configures the linearizer at the current trial state
        # before asking the inverter to apply the operator.
        self.u = state.value

    @staticmethod
    @ACCELERATOR.compile
    def jacobian_apply(u, translation_u, laplacian_u, result_u, diffusivity):
        result_u[:] = diffusivity * laplacian_u + (1.0 - 3.0 * u * u) * translation_u

    def __call__(
        self,
        translation: StarkVectorTranslation,
        out: StarkVectorTranslation,
    ) -> None:
        u = self.u
        assert u is not None
        AllenCahnRHS.laplacian_periodic(
            translation.value,
            self.laplacian_u,
            self.inv_dx2,
        )
        self.jacobian_apply(
            u,
            translation.value,
            self.laplacian_u,
            out.value,
            self.diffusivity,
        )


class AllenCahnLinearizer:
    __slots__ = ("operator",)
    _compiled = False

    def __init__(self, geometry: Geometry, diffusivity: float) -> None:
        self.operator = AllenCahnJacobianOperator(geometry, diffusivity)

        if not self.__class__._compiled:
            probe = np.zeros(geometry.grid_size, dtype=np.float64)
            ACCELERATOR.compile_examples(
                AllenCahnJacobianOperator.jacobian_apply,
                (probe, probe, probe, probe, diffusivity),
            )
            self.__class__._compiled = True

    def __call__(self, interval: Interval, state: StarkVector, out) -> None:
        del interval
        # `out` is STARK's operator probe. Setting `out.apply` gives the Newton
        # resolvent a callable matrix-free linear operator.
        self.operator.configure(state)
        out.apply = self.operator


def allen_cahn_inner_product(
    left: StarkVectorTranslation,
    right: StarkVectorTranslation,
) -> float:
    return float(np.dot(left.value, right.value))


if __name__ == "__main__":
    geometry = Geometry()
    executor = Executor(tolerance=EXECUTOR_TOLERANCE)

    # Start from the high-level interface again, then keep the prepared
    # derivative and allocator. Fully implicit methods need those lower-level
    # pieces because the resolvent and inverter work in translation space.

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
    allocator = template.allocator

    # BiCGStab is a matrix-free Krylov inverter. It needs an inner product on
    # translations; for a NumPy grid field we use the ordinary dot product.

    linearizer = AllenCahnLinearizer(geometry, DIFFUSIVITY)
    inverter = InverterBiCGStab(
        allocator,
        allen_cahn_inner_product,
        ExecutorTolerance=InverterTolerance(atol=1.0e-7, rtol=1.0e-7),
        policy=InverterPolicy(max_iterations=24, restart=12),
        safety=executor.safety,
    )

    # `ResolventNewton` owns the nonlinear solve for each implicit stage.
    # `SchemeSDIRK21.tableau` tells the resolvent which stage equation it is
    # being asked to solve.

    resolvent = ResolventNewton(
        allocator,
        linearizer=linearizer,
        inverter=inverter,
        ExecutorTolerance=ResolventTolerance(atol=1.0e-7, rtol=1.0e-7),
        policy=ResolventPolicy(max_iterations=12),
        safety=executor.safety,
        accelerator=ACCELERATOR,
        tableau=SchemeSDIRK21.tableau,
    )

    implicit_scheme = SchemeSDIRK21(derivative, allocator, resolvent)
    explicit_scheme = SchemeCashKarp(derivative, allocator)

    # Compare against the explicit method rather than presenting Newton in
    # isolation. The result is usually instructive: fewer accepted time steps do
    # not automatically mean a faster solve when each step contains nonlinear
    # and linear iterations.

    problem = ComparisonProblem(
        name="Allen-Cahn implicit Newton",
        build_state=lambda: StarkVector(initial_profile(geometry), carrier),
        build_interval=make_interval,
        difference=state_difference,
        diagnostics=state_diagnostics,
    )

    entries = [
        ComparisonEntry("SDIRK21 Newton", Marcher(implicit_scheme, executor)),
        ComparisonEntry("Cash-Karp", Marcher(explicit_scheme, executor)),
    ]

    # The interesting part of this report is the tradeoff. SDIRK21 may take
    # fewer accepted steps, but each accepted step contains Newton iterations,
    # and each Newton iteration contains linear inverter work. The profile
    # breakdown makes that cost visible.

    report = ComparisonRunner(problem, entries, repeats=3)()
    print(report)
    print()
    print("What to notice:")
    print("- Fewer time steps do not automatically mean a faster method.")
    print("- The profile breakdown shows whether time moved into resolvent/inverter work.")
    print("- This motivates the IMEX split in the next lesson.")
