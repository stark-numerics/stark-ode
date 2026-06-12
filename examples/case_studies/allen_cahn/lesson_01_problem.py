from __future__ import annotations

# Lesson 1: problem definition and first System solve
#
# This script is the foundation for the Allen-Cahn example series. The problem
# state is a NumPy array, so the modern STARK entry point is the interface layer:
# `System` prepares the matching layout-backed state, allocator,
# derivative adapter, and carrier routing for us.
#
# We solve the one-dimensional periodic Allen-Cahn equation
#
#     u_t = D u_xx + u - u^3.
#
# There are two natural parts:
#
# - diffusion: D u_xx
# - reaction:  u - u^3
#
# In this first lesson we treat the whole right-hand side explicitly with an
# adaptive Cash-Karp method. Later lessons compare explicit methods, inspect
# monitoring data, drop down to the engine boundary for implicit solves,
# and then split the PDE into an IMEX method with a custom spectral resolvent.
#
# The point of this first run is not to claim Cash-Karp is the right Allen-Cahn
# solver. It is to establish a baseline: with `System`, an array-valued PDE
# can be integrated without hand-writing STARK state, translation, and allocator
# classes.
#
# In a source checkout, run from the `stark-ode` directory with:
#
#     python -m examples.case_studies.allen_cahn.lesson_01_problem

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

# Write figures beside the lesson instead of opening a GUI window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from stark import Configuration, Interval, Layout, Method, System, Tolerance
from stark.engines.accelerators import AcceleratorNone, AcceleratorNumba
from stark.engines import EngineNumpy
from stark.methods.schemes import SchemeCashKarp


# We keep the numerical choices near the top so later lessons can import them
# without hiding the scale of the experiment. The default grid is intentionally
# modest: large enough to feel like a PDE, small enough for an example run.


HERE = Path(__file__).resolve().parent

DIFFUSIVITY: float = 0.08
Configuration_TOLERANCE = Tolerance(atol=1.0e-6, rtol=1.0e-3)
START_TIME = 0.0
STOP_TIME = 1.0
INITIAL_STEP = 1.0e-3

try:
    ACCELERATOR = AcceleratorNumba()
except ModuleNotFoundError:
    # Numba is useful for this grid example, but the lesson should still run in
    # an environment with only the core STARK dependencies available.
    ACCELERATOR = AcceleratorNone()


# The geometry object carries the periodic cell length, grid size, spacing, and
# grid points. We keep it tiny because the finite-difference derivative will ask
# for only these pieces of information.


@dataclass(slots=True)
class Geometry:
    length: float = 6.0
    grid_size: int = 364
    _dx: float | None = field(init=False, repr=True, default=None)
    _x: np.ndarray | None = field(init=False, repr=False, default=None)

    @property
    def dx(self) -> float:
        dx = self._dx
        if dx is None:
            dx = self.length / self.grid_size
            self._dx = dx
        return dx

    @property
    def x(self) -> np.ndarray:
        x = self._x
        if x is None:
            x = np.linspace(
                0.0,
                self.length,
                self.grid_size,
                endpoint=False,
                dtype=np.float64,
            )
            self._x = x
        return x


def initial_profile(geometry: Geometry) -> np.ndarray:
    # A smooth multi-mode initial condition is enough to show diffusion,
    # reaction, and time-step adaptation without introducing discontinuities.
    x = geometry.x
    k = 2.0 * np.pi / geometry.length
    u = 0.5 * np.sin(k * x) + 0.5 * np.sin(3.0 * k * x)
    return u.astype(np.float64)


def make_interval() -> Interval:
    # STARK expects an explicit interval object. It deliberately does not
    # accept a bare tuple, because the solver mutates `present` and `step`.
    return Interval(
        present=START_TIME,
        step=INITIAL_STEP,
        stop=STOP_TIME,
    )


def profile_diagnostics(profile: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(profile.mean()),
        "max": float(profile.max()),
        "min": float(profile.min()),
    }


def state_difference(left, right) -> float:
    # ComparisonRunner examples work with final STARK states, so the difference
    # function receives layout-backed objects and looks at their `u` fields.
    diff = left.u - right.u
    return float((diff @ diff / diff.size) ** 0.5)


def state_diagnostics(state) -> dict[str, float]:
    return profile_diagnostics(state.u)


# The derivative kernel works on raw NumPy arrays. `make_derivative(...)` adapts
# it to the layout-backed objects that `System` gives to schemes.


class AllenCahnRHS:
    _compiled = False

    def __init__(self, geometry: Geometry, diffusivity: float) -> None:
        self.geometry = geometry
        self.diffusivity = diffusivity
        self.inv_dx2 = 1.0 / (geometry.dx * geometry.dx)
        self.laplacian_u = np.zeros(geometry.grid_size, dtype=np.float64)

        if not self.__class__._compiled:
            probe = np.zeros(geometry.grid_size, dtype=np.float64)
            ACCELERATOR.compile_examples(
                self.laplacian_periodic,
                (probe, probe, self.inv_dx2),
            )
            ACCELERATOR.compile_examples(
                self.full_rhs,
                (probe, probe, probe, self.diffusivity),
            )
            self.__class__._compiled = True

    def __call__(self, time: float, u: np.ndarray, out: np.ndarray) -> None:
        del time
        self.laplacian_periodic(u, self.laplacian_u, self.inv_dx2)
        self.full_rhs(u, self.laplacian_u, out, self.diffusivity)

    @staticmethod
    @ACCELERATOR.compile
    def laplacian_periodic(field, out, inv_dx2):
        size = field.size
        for index in range(size):
            left = field[index - 1 if index > 0 else size - 1]
            centre = field[index]
            right = field[index + 1 if index + 1 < size else 0]
            out[index] = (left - 2.0 * centre + right) * inv_dx2

    @staticmethod
    @ACCELERATOR.compile
    def full_rhs(u, laplacian_u, out_u, diffusivity):
        out_u[:] = diffusivity * laplacian_u + u - u * u * u

    @staticmethod
    @ACCELERATOR.compile
    def reaction_rhs(u, out_u):
        out_u[:] = u - u * u * u

    @staticmethod
    @ACCELERATOR.compile
    def diffusion_rhs(laplacian_u, out_u, diffusivity):
        out_u[:] = diffusivity * laplacian_u


def make_derivative(geometry: Geometry, diffusivity: float = DIFFUSIVITY):
    rhs = AllenCahnRHS(geometry, diffusivity)

    def derivative(time: float, state, out) -> None:
        rhs(time, state.u, out.du)

    return derivative


def make_layout(geometry: Geometry) -> Layout:
    return Layout({"u": {"translation": "du", "shape": (geometry.grid_size,)}})


def make_system(geometry: Geometry, diffusivity: float = DIFFUSIVITY) -> System:
    return System(
        derivative=make_derivative(geometry, diffusivity),
        layout=make_layout(geometry),
    )


def make_ivp(
    geometry: Geometry,
    *,
    method: Method | None = None,
    configuration: Configuration | None = None,
    initial: np.ndarray | None = None,
    interval: Interval | None = None,
):
    return make_system(geometry).ivp(
        initial={"u": initial_profile(geometry) if initial is None else initial},
        interval=make_interval() if interval is None else interval,
        method=Method(scheme=SchemeCashKarp) if method is None else method,
        engine=EngineNumpy,
        configuration=Configuration(scheme_tolerance=Configuration_TOLERANCE) if configuration is None else configuration,
    )


if __name__ == "__main__":
    geometry = Geometry()
    initial = initial_profile(geometry)
    diagnostics = profile_diagnostics(initial)

    print(f"initial mean: {diagnostics['mean']:.4f}")
    print(f"initial max:  {diagnostics['max']:.4f}")
    print(f"initial min:  {diagnostics['min']:.4f}")

    # First look at the initial condition before involving STARK. This is just
    # ordinary user-side plotting of a NumPy array.

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(geometry.x, initial)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title("Allen-Cahn initial condition")
    initial_plot_path = HERE / "allen_cahn_initial_condition.png"
    fig.savefig(initial_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved {initial_plot_path}")

    # `System` is the high-level path for shaped NumPy states. The layout
    # names the state field `u` and its matching translation `du`; the engine
    # owns the carrier, allocator, and generated algebra.

    ivp = make_ivp(geometry, initial=initial)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(geometry.x, initial, label=f"t={START_TIME:g}")

    # Stable trajectory mode yields copied snapshots at the requested
    # checkpoints, which is the right default for plotting.

    for snapshot_interval, snapshot_state in ivp.stable_trajectory(
        checkpoints=5,
    ):
        diagnostics = state_diagnostics(snapshot_state)
        print(
            f"t: {snapshot_interval.present:.4f}, "
            f"mean: {diagnostics['mean']:.4f}, "
            f"max: {diagnostics['max']:.4f}, "
            f"min: {diagnostics['min']:.4f}"
        )
        ax.plot(
            geometry.x,
            snapshot_state.u,
            label=f"t={snapshot_interval.present:.2f}",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title("Cash-Karp Allen-Cahn profiles")
    ax.legend(loc="upper right")
    profile_plot_path = HERE / "allen_cahn_cash_karp_profiles.png"
    fig.savefig(profile_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved {profile_plot_path}")
    print()
    print("What to notice:")
    print("- The mean stays near zero because the initial condition is symmetric.")
    print("- The extrema change smoothly as diffusion and reaction compete.")
    print("- We reached this first result through System, not custom adapter classes.")
