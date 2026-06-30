from __future__ import annotations

from typing import Any

import numpy as np

from competition.hires import common

jax: Any
dfx: Any
eqx: Any
jnp: Any

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import diffrax as dfx
    import equinox as eqx
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency for local use
    jax = dfx = eqx = jnp = None
    DIFFRAX_AVAILABLE = False
else:
    DIFFRAX_AVAILABLE = True


def vector_field(t, y, args):
    del t, args
    y0, y1, y2, y3, y4, y5, y6, y7 = y
    reaction = 280.0 * y5 * y7
    return jnp.array(
        [
            -1.71 * y0 + 0.43 * y1 + 8.32 * y2 + 0.0007,
            1.71 * y0 - 8.75 * y1,
            -10.03 * y2 + 0.43 * y3 + 0.035 * y4,
            8.32 * y1 + 1.71 * y2 - 1.12 * y3,
            -1.745 * y4 + 0.43 * y5 + 0.43 * y6,
            -reaction + 0.69 * y3 + 1.71 * y4 - 0.43 * y5 + 0.69 * y6,
            reaction - 1.81 * y6,
            -reaction + 1.81 * y6,
        ],
        dtype=jnp.float64,
    )


def prepare_kvaerno5(problem_parameters, tolerance_parameters, initial_conditions, reference):
    if not DIFFRAX_AVAILABLE:
        raise RuntimeError("Diffrax/JAX is not installed.")

    term = dfx.ODETerm(vector_field)
    solver = dfx.Kvaerno5()
    saveat = dfx.SaveAt(t1=True)
    stepsize_controller = dfx.PIDController(
        rtol=tolerance_parameters["rtol"],
        atol=tolerance_parameters["atol"],
    )
    y0 = jnp.asarray(initial_conditions["y"])

    @eqx.filter_jit
    def solve(y0):
        return dfx.diffeqsolve(
            term,
            solver,
            t0=problem_parameters["t0"],
            t1=problem_parameters["t1"],
            dt0=common.DIFFRAX_PARAMETERS["dt0"],
            y0=y0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=1_000_000,
        )

    def solve_once():
        solution = solve(y0)
        jax.block_until_ready(solution.ys)
        final = np.asarray(solution.ys[0])
        error = np.sqrt(np.dot(final - reference["y"], final - reference["y"]) / final.size)
        return {
            "library": "Diffrax",
            "solver": "Kvaerno5",
            "error": float(error),
            "steps": int(solution.stats["num_steps"]),
        }

    return solve_once
