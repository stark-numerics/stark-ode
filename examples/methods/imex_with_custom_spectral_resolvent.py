"""Use IMEX splitting with a custom spectral resolvent.

Some stiff problems have structure a general Newton solve should not ignore.
Here the nonlinear reaction is explicit, while periodic linear diffusion is
implicit and solved directly in Fourier space by a small custom resolvent.
"""

from __future__ import annotations

import numpy as np

from stark import Configuration, Derivative, Frame, Interval, Method, System, Tolerance
from stark.engines import EngineNumpy
from stark.methods import SchemeKennedyCarpenter43_7


GRID_SIZE = 64
LENGTH = 6.0
DIFFUSIVITY = 0.08


def initial_profile() -> np.ndarray:
    x = np.linspace(0.0, LENGTH, GRID_SIZE, endpoint=False)
    wave_number = 2.0 * np.pi / LENGTH
    return 0.5 * np.sin(wave_number * x) + 0.5 * np.sin(3.0 * wave_number * x)


def laplacian_periodic(field: np.ndarray, out: np.ndarray) -> None:
    dx = LENGTH / GRID_SIZE
    out[:] = (np.roll(field, 1) - 2.0 * field + np.roll(field, -1)) / (dx * dx)


def implicit_diffusion(_time: float, state, out) -> None:
    laplacian_periodic(state.u, out.du)
    out.du[:] *= DIFFUSIVITY


def explicit_reaction(_time: float, state, out) -> None:
    out.du[:] = state.u - state.u * state.u * state.u


class SpectralDiffusionResolvent:
    """Directly solve the implicit periodic diffusion stage in Fourier space."""

    def __init__(self) -> None:
        theta = 2.0 * np.pi * np.fft.fftfreq(GRID_SIZE)
        inv_dx2 = 1.0 / ((LENGTH / GRID_SIZE) * (LENGTH / GRID_SIZE))
        self.symbol = DIFFUSIVITY * 2.0 * (np.cos(theta) - 1.0) * inv_dx2
        self.spectrum = np.zeros(GRID_SIZE, dtype=np.complex128)

    def __call__(self, problem, out) -> None:
        base_u = problem.origin.u
        rhs = problem.rhs
        alpha = problem.alpha

        self.spectrum[:] = np.fft.fft(base_u)
        self.spectrum[:] *= alpha * self.symbol
        if rhs is not None:
            self.spectrum[:] += np.fft.fft(rhs[0].du)

        self.spectrum[:] /= 1.0 - alpha * self.symbol
        out[0].du[:] = np.fft.ifft(self.spectrum).real


if __name__ == "__main__":
    derivative = Derivative.split(
        implicit=Derivative(implicit_diffusion),
        explicit=Derivative(explicit_reaction),
    )
    system = System(
        derivative=derivative,
        frame=Frame.array("u", translation="du", shape=(GRID_SIZE,)),
    )
    ivp = system.ivp(
        initial={"u": initial_profile()},
        interval=Interval(present=0.0, step=1.0e-3, stop=0.1),
        method=Method(
            SchemeKennedyCarpenter43_7,
            resolvent=SpectralDiffusionResolvent(),
        ),
        engine=EngineNumpy,
        configuration=Configuration(
            scheme_tolerance=Tolerance(atol=1.0e-6, rtol=1.0e-3),
        ),
    )

    print("IMEX with a custom spectral diffusion resolvent")
    for interval, state in ivp.stable_trajectory(checkpoints=4):
        print(
            f"t={interval.present:.3f}, "
            f"mean={state.u.mean(): .6f}, "
            f"min={state.u.min(): .6f}, "
            f"max={state.u.max(): .6f}"
        )
