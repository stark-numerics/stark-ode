from __future__ import annotations

# Lesson 7: larger IMEX visualisation run
#
# At the smaller grid size, the IMEX method has shown that the split is not
# just mathematically neat, but computationally worthwhile. For a final
# demonstration, we now increase the size of the spatial grid and use the IMEX
# stepper to generate a higher-resolution solution.
#
# The goal here is no longer to compare methods, but to use the method that has
# emerged from the story and let it produce a clean picture of the Allen-Cahn
# dynamics. We record the evolving field over time and visualise both:
#
# - the full space-time evolution,
# - and the initial and final profiles.
#
# This is a richer visual run rather than a cheap smoke-test candidate.
#
# In a source checkout, run from the `stark-ode` directory with:
#
#     python -m examples.case_studies.allen_cahn.lesson_07_large_imex_run

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from stark import Configuration, Integrator, Interval, IntegratorStepper, Method, Tolerance
from stark.core.contracts import DerivativeIMEX
from stark.methods.schemes import SchemeCashKarp, SchemeKennedyCarpenter43_7

from examples.case_studies.allen_cahn.lesson_01_problem import (
    DIFFUSIVITY,
    Geometry,
    initial_profile,
    make_ivp,
)
from examples.case_studies.allen_cahn.lesson_05_imex_spectral import (
    AllenCahnExplicitDerivative,
    AllenCahnImplicitDerivative,
    AllenCahnSpectralResolvent,
)


HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    geometry = Geometry(grid_size=1024)
    configuration_tolerance = Tolerance(atol=1.0e-6, rtol=1.0e-3)
    start_time = 0.0
    stop_time = 5.0
    initial_step = 1.5e-3
    configuration = Configuration(scheme_tolerance=configuration_tolerance)

    # We still let `System` prepare the carrier and allocator, even though
    # the solve itself uses a hand-assembled IMEX scheme.

    template = make_ivp(
        geometry,
        method=Method(scheme=SchemeCashKarp),
        configuration=configuration,
        interval=Interval(present=start_time, step=initial_step, stop=stop_time),
    )

    implicit_derivative = AllenCahnImplicitDerivative(geometry, DIFFUSIVITY)
    explicit_derivative = AllenCahnExplicitDerivative(geometry)
    derivative = DerivativeIMEX(
        implicit=implicit_derivative,
        explicit=explicit_derivative,
    )

    allocator = template.engine.allocator
    resolvent = AllenCahnSpectralResolvent(geometry, DIFFUSIVITY)

    scheme = SchemeKennedyCarpenter43_7(
        derivative,
        allocator,
        resolvent=resolvent,
    )
    integrate = Integrator(configuration=configuration)
    stepper = IntegratorStepper(scheme)

    initial = initial_profile(geometry)
    state = template.fresh_state()
    interval = Interval(present=start_time, step=initial_step, stop=stop_time)
    checkpoints = np.linspace(start_time, stop_time, 120)[1:]

    # Store raw arrays for plotting, not live STARK state objects. The solver
    # mutates its state in place as it advances.

    times = [start_time]
    frames = [state.u.copy()]

    for snapshot_interval, snapshot_state in integrate(
        stepper,
        interval,
        state,
        checkpoints=checkpoints,
    ):
        times.append(float(snapshot_interval.present))
        frames.append(snapshot_state.u.copy())

    times = np.asarray(times)
    frames = np.asarray(frames)
    x = geometry.x

    # The top panel shows the full space-time evolution. The lower panel keeps
    # the simple before/after view from the original notebook.

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

    image = axes[0].imshow(
        frames,
        aspect="auto",
        cmap="coolwarm",
        extent=(0.0, geometry.length, times[-1], times[0]),
    )
    axes[0].set_title("Allen-Cahn evolution with IMEX spectral solve")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    fig.colorbar(image, ax=axes[0], label="u")

    axes[1].plot(x, initial, linewidth=1.8, label="initial")
    axes[1].plot(x, frames[-1], linewidth=2.2, label="final")
    axes[1].set_title("Initial and final profile")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u")
    axes[1].legend(loc="upper right")

    plot_path = HERE / "allen_cahn_imex_evolution.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved {plot_path}")
    print()
    print("What to notice:")
    print("- The heatmap shows the full space-time history, not just final diagnostics.")
    print("- The final profile shows the same Allen-Cahn coarsening trend in a compact view.")
    print("- This lesson is visual validation; keep smaller lessons for cheap smoke tests.")
    print()
    print("Conclusion:")
    print("- Start with the simplest working model.")
    print("- Test it, compare it, and inspect how it behaves.")
    print("- Add specialised machinery only when the problem justifies the extra structure.")
    print("- For Allen-Cahn, that evidence points toward the IMEX spectral method.")
