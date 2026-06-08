from __future__ import annotations

# Lesson 5: IMEX split with a custom spectral resolvent
#
# In Allen-Cahn, the reaction term is nonlinear and local, while diffusion is
# linear. That suggests an IMEX split:
#
#     f_implicit(u) = D L u
#     f_explicit(u) = u - u^3
#
# where L is the periodic finite-difference Laplacian. On a periodic grid, the
# Laplacian is diagonal in Fourier space. We can therefore write a custom
# resolvent for the implicit diffusion stage and avoid the expensive Newton
# solve from lesson 4.
#
# In a source checkout, run from the `stark-ode` directory with:
#
#     python -m examples.case_studies.allen_cahn.lesson_05_imex_spectral
#
# The point is to keep the stability benefit of implicit diffusion while
# avoiding a general nonlinear solve. Because the diffusion operator is linear
# and diagonal in Fourier space, the implicit stage solve can be direct and
# problem-specific.

import numpy as np

from stark import Configuration, Integrator, Interval, IntegratorStepper, StarkMethod
from stark.contracts import DerivativeIMEX
from stark.schemes import SchemeCashKarp, SchemeKennedyCarpenter43_7

from examples.case_studies.allen_cahn.lesson_01_problem import (
    DIFFUSIVITY,
    Configuration_TOLERANCE,
    AllenCahnRHS,
    Geometry,
    make_ivp,
    state_diagnostics,
)


# The implicit part is only diffusion. The explicit part is only reaction. Both
# still satisfy the same derivative contract: fill a translation in place.
#
# With the split
#
#     f_im(u) = D L u,
#     f_ex(u) = u - u^3,
#
# the nonlinear reaction term stays on the explicit side. By the time an
# implicit stage is solved, the reaction contribution is already known. The
# unknown part is only the linear diffusion solve.


class AllenCahnImplicitDerivative:
    __slots__ = ("diffusivity", "inv_dx2", "laplacian_u")

    def __init__(self, geometry: Geometry, diffusivity: float) -> None:
        self.diffusivity = diffusivity
        self.inv_dx2 = 1.0 / (geometry.dx * geometry.dx)
        self.laplacian_u = np.zeros(geometry.grid_size, dtype=np.float64)

    def __call__(
        self,
        interval: Interval,
        state,
        out,
    ) -> None:
        del interval
        AllenCahnRHS.laplacian_periodic(
            state.u,
            self.laplacian_u,
            self.inv_dx2,
        )
        out.du[:] = self.diffusivity * self.laplacian_u


class AllenCahnExplicitDerivative:
    __slots__ = ()

    def __init__(self, geometry: Geometry) -> None:
        del geometry

    def __call__(
        self,
        interval: Interval,
        state,
        out,
    ) -> None:
        del interval
        out.du[:] = state.u - state.u**3


class AllenCahnSpectralResolvent:
    def __init__(self, geometry: Geometry, diffusivity: float) -> None:
        # For the periodic second-difference operator, each Fourier mode is an
        # eigenvector. `operator_symbol` stores the corresponding eigenvalues.
        theta = 2.0 * np.pi * np.fft.fftfreq(geometry.grid_size)
        inv_dx2 = 1.0 / (geometry.dx * geometry.dx)
        self.operator_symbol = diffusivity * 2.0 * (np.cos(theta) - 1.0) * inv_dx2
        self.spectrum = np.zeros(geometry.grid_size, dtype=np.complex128)

    def bind(self, interval: Interval, state) -> None:
        # The current resolvent contract passes the stage problem directly to
        # `__call__`; this method remains as a no-op for older display text and
        # custom resolvent examples that still mention binding.
        del interval, state

    def __call__(self, problem, out) -> None:
        base_u = problem.origin.u
        alpha = problem.alpha
        rhs = problem.rhs

        # The stage equation presented by STARK is
        #
        #     delta - rhs - alpha D L(base + delta) = 0.
        #
        # In Fourier space this becomes one scalar solve per mode:
        #
        #     (1 - alpha lambda_k) delta_hat_k
        #         = alpha lambda_k base_hat_k + rhs_hat_k.
        #
        # That is why this split is attractive: no nonlinear iteration and no
        # Krylov inverter are needed inside the stage solve.

        self.spectrum[:] = np.fft.fft(base_u)
        self.spectrum[:] *= alpha * self.operator_symbol

        if rhs is not None:
            self.spectrum[:] += np.fft.fft(rhs[0].du)

        self.spectrum[:] /= 1.0 - alpha * self.operator_symbol
        out[0].du[:] = np.fft.ifft(self.spectrum).real


if __name__ == "__main__":
    # The display helper is intentionally printed before the implementation is
    # used. It states the resolvent contract this custom spectral solver must
    # satisfy.

    print("Implicit stage problem for KC43-7")
    print("---------------------------------")
    print(SchemeKennedyCarpenter43_7.display_resolvent_problem())

    geometry = Geometry()
    configuration = Configuration(scheme_tolerance=Configuration_TOLERANCE)

    # `StarkSystem` prepares the carrier/allocator machinery. We then replace
    # the derivative with an IMEX pair because this lesson is about the split,
    # not about reimplementing carriers by hand.

    template = make_ivp(
        geometry,
        method=StarkMethod(scheme=SchemeCashKarp),
        configuration=configuration,
    )

    implicit_derivative = AllenCahnImplicitDerivative(geometry, DIFFUSIVITY)
    explicit_derivative = AllenCahnExplicitDerivative(geometry)
    derivative = DerivativeIMEX(
        implicit=implicit_derivative,
        explicit=explicit_derivative,
    )

    allocator = template.engine.allocator
    resolvent = AllenCahnSpectralResolvent(geometry, DIFFUSIVITY)

    # Kennedy-Carpenter 4(3) 7-stage is an adaptive IMEX method. The explicit
    # reaction stages are ordinary RHS evaluations; the implicit diffusion
    # stages call our spectral resolvent.

    scheme = SchemeKennedyCarpenter43_7(
        derivative,
        allocator,
        resolvent=resolvent,
    )
    integrate = Integrator(configuration=configuration)
    stepper = IntegratorStepper(scheme)

    interval = template.fresh_interval()
    state = template.fresh_state()

    # Snapshot integration copies the state at each checkpoint. That is the
    # safer choice when the caller wants to keep all yielded states.

    for snapshot_interval, snapshot_state in integrate(
        stepper,
        interval,
        state,
        checkpoints=5,
    ):
        diagnostics = state_diagnostics(snapshot_state)
        print(
            f"t: {snapshot_interval.present:.4f}, "
            f"mean: {diagnostics['mean']:.4f}, "
            f"max: {diagnostics['max']:.4f}, "
            f"min: {diagnostics['min']:.4f}"
        )
    print()
    print("What to notice:")
    print("- The checkpoint diagnostics match the explicit baseline closely.")
    print("- The implicit work is now a direct spectral solve, not Newton plus Krylov.")
    print("- This is the payoff for splitting the PDE according to its structure.")
