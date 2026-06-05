from __future__ import annotations

# Lesson 6: compare explicit, fully implicit, and IMEX methods
#
# At this point, a single IMEX run only tells us that the custom resolvent
# works. The more interesting question is whether it actually improves the
# solve.
#
# So we now compare three approaches on the same Allen-Cahn problem:
#
# - an adaptive explicit method,
# - an adaptive fully implicit method with Newton,
# - and the custom IMEX method with the spectral diffusion resolvent.
#
# This is the decision point in the story. We have added quite a lot of
# machinery since lesson 1, and this comparison asks whether that extra
# structure bought us a better balance between step count and internal cost.
#
# In a source checkout, run from the `stark-ode` directory with:
#
#     python -m examples.case_studies.allen_cahn.lesson_06_compare_methods

from stark import IntegratorStepper, Tolerance
from stark.comparison import ComparisonRunner, ComparisonEntry, ComparisonProblem
from stark.contracts import DerivativeIMEX
from stark.interface import StarkDerivative, StarkIVP, StarkVector
from stark.inverters import InverterBiCGStab, InverterPolicy
from stark.resolvents import ResolventNewton
from stark.schemes import SchemeCashKarp, SchemeKennedyCarpenter43_7, SchemeSDIRK21

from examples.case_studies.allen_cahn.lesson_01_problem import (
    ACCELERATOR,
    DIFFUSIVITY,
    Configuration_TOLERANCE,
    AllenCahnRHS,
    Geometry,
    initial_profile,
    make_interval,
    state_diagnostics,
    state_difference,
)
from examples.case_studies.allen_cahn.lesson_04_implicit_newton import (
    AllenCahnLinearizer,
    allen_cahn_inner_product,
)
from examples.case_studies.allen_cahn.lesson_05_imex_spectral import (
    AllenCahnExplicitDerivative,
    AllenCahnImplicitDerivative,
    AllenCahnSpectralResolvent,
)


if __name__ == "__main__":
    geometry = Geometry()
    configuration = Configuration(scheme_tolerance=Configuration_TOLERANCE)

    # Prepare the vector-space boundary once. All three methods below solve the
    # same semidiscrete Allen-Cahn problem on the same carrier.

    template = StarkIVP(
        derivative=StarkDerivative.in_place(
            AllenCahnRHS(geometry, DIFFUSIVITY),
        ),
        initial=initial_profile(geometry),
        interval=make_interval(),
        scheme=SchemeCashKarp,
        configuration=Configuration,
    ).build()

    carrier = template.initial.carrier
    full_derivative = template.derivative
    allocator = template.allocator

    # 1. Explicit baseline: simple to configure, but lesson 3 showed that it
    # may take conservative steps on this problem.

    explicit_scheme = SchemeCashKarp(full_derivative, allocator)

    # 2. Fully implicit Newton: fewer macro steps may be possible, but each
    # step carries nonlinear and Krylov linear-solve work.

    linearizer = AllenCahnLinearizer(geometry, DIFFUSIVITY)
    inverter = InverterBiCGStab(
        allocator,
        allen_cahn_inner_product,
        configuration=Configuration(inverter_tolerance=Tolerance(atol=1.0e-7, rtol=1.0e-7)),
        policy=InverterPolicy(max_iterations=24, restart=12),
    )
    newton_resolvent = ResolventNewton(
        allocator,
        linearizer=linearizer,
        inverter=inverter,
        configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-7, rtol=1.0e-7), resolvent_maximum_steps=12),
        accelerator=ACCELERATOR,
        tableau=SchemeSDIRK21.tableau,
    )
    implicit_scheme = SchemeSDIRK21(
        full_derivative,
        allocator,
        newton_resolvent,
    )

    # 3. IMEX spectral: treat only linear diffusion implicitly and solve that
    # stage problem directly in Fourier space.

    imex_derivative = DerivativeIMEX(
        implicit=AllenCahnImplicitDerivative(geometry, DIFFUSIVITY),
        explicit=AllenCahnExplicitDerivative(geometry),
    )
    spectral_resolvent = AllenCahnSpectralResolvent(geometry, DIFFUSIVITY)
    imex_scheme = SchemeKennedyCarpenter43_7(
        imex_derivative,
        allocator,
        resolvent=spectral_resolvent,
    )

    problem = ComparisonProblem(
        name="Allen-Cahn method comparison",
        build_state=lambda: StarkVector(initial_profile(geometry), carrier),
        build_interval=make_interval,
        difference=state_difference,
        diagnostics=state_diagnostics,
    )

    entries = [
        ComparisonEntry("Cash-Karp explicit", IntegratorStepper(explicit_scheme)),
        ComparisonEntry("SDIRK21 Newton", IntegratorStepper(implicit_scheme)),
        ComparisonEntry("KC43-7 IMEX spectral", IntegratorStepper(imex_scheme)),
    ]

    report = ComparisonRunner(problem, entries, repeats=3)()
    print(report)
    print()
    print("What to notice:")
    print("- The explicit method is the simplest baseline.")
    print("- The fully implicit method shows what generic nonlinear solving costs.")
    print("- The IMEX spectral method tests whether problem structure pays for itself.")
    print("- The best method here is not chosen by taste; it is chosen by evidence.")
