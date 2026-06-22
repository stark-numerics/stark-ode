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

from stark import Configuration, Derivative, IntegratorStepper, Method, Tolerance
from stark.diagnostics.comparison import ComparisonRunner, ComparisonEntryStepper, ComparisonProblem
from stark.methods.inverters import InverterKrylovArnoldi
from stark.methods.resolvents import ResolventNewton
from stark.methods.schemes import SchemeCashKarp, SchemeKennedyCarpenter43_7, SchemeSDIRK21

from examples.case_studies.allen_cahn.lesson_01_problem import (
    ACCELERATOR,
    DIFFUSIVITY,
    Configuration_TOLERANCE,
    Geometry,
    make_ivp,
    state_diagnostics,
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

    # Prepare the frame-backed boundary once. All three methods below solve
    # the same semidiscrete Allen-Cahn problem on the same engine allocator.

    template = make_ivp(
        geometry,
        method=Method(scheme=SchemeCashKarp),
        configuration=configuration,
    )

    full_derivative = template.scheme.derivative
    allocator = template.engine.allocator

    # 1. Explicit baseline: simple to configure, but lesson 3 showed that it
    # may take conservative steps on this problem.

    explicit_scheme = SchemeCashKarp(full_derivative, allocator)

    # 2. Fully implicit Newton: fewer macro steps may be possible, but each
    # step carries nonlinear and Krylov linear-solve work. The current Arnoldi
    # inverter uses the same request-shaped inverter protocol as the dense and
    # relaxation families.

    linearizer = AllenCahnLinearizer(geometry, DIFFUSIVITY)
    inverter = InverterKrylovArnoldi(
        allocator,
        allen_cahn_inner_product,
        restart=12,
        configuration=Configuration(
            inverter_tolerance=Tolerance(atol=1.0e-7, rtol=1.0e-7),
            inverter_maximum_steps=24,
        ),
        accelerator=ACCELERATOR,
    )
    newton_resolvent = ResolventNewton(
        allocator,
        linearizer=linearizer,
        inverter=inverter,
        configuration=Configuration(
            resolvent_tolerance=Tolerance(atol=1.0e-7, rtol=1.0e-7),
            resolvent_maximum_steps=12,
        ),
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

    imex_derivative = Derivative.split(
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
        "Allen-Cahn method comparison",
        template,
        diagnostics=state_diagnostics,
    )

    entries = [
        ComparisonEntryStepper("Cash-Karp explicit", IntegratorStepper(explicit_scheme)),
        ComparisonEntryStepper("SDIRK21 Newton", IntegratorStepper(implicit_scheme)),
        ComparisonEntryStepper("KC43-7 IMEX spectral", IntegratorStepper(imex_scheme)),
    ]

    report = ComparisonRunner(problem, entries, repeats=3)()
    print(report)
    print()
    print("What to notice:")
    print("- The explicit method is the simplest baseline.")
    print("- The fully implicit method shows what generic nonlinear solving costs.")
    print("- The IMEX spectral method tests whether problem structure pays for itself.")
    print("- The best method here is not chosen by taste; it is chosen by evidence.")
