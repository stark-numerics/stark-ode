from __future__ import annotations

import numpy as np

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import diffrax as dfx
    import equinox as eqx
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency for local use
    DIFFRAX_AVAILABLE = False
else:
    DIFFRAX_AVAILABLE = True


def vector_field(t, y, args):
    del t
    y1, y2, y3 = y
    return jnp.array(
        [
            -0.04 * y1 + 1.0e4 * y2 * y3,
            0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2,
            3.0e7 * y2 * y2,
        ],
        dtype=jnp.float64,
    )


def prepare_kvaerno5(problem_parameters, tolerance_parameters, diffrax_parameters, initial_conditions, reference):
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
            dt0=diffrax_parameters["dt0"],
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


def run_kvaerno5(problem_parameters, tolerance_parameters, diffrax_parameters, initial_conditions, reference):
    return prepare_kvaerno5(problem_parameters, tolerance_parameters, diffrax_parameters, initial_conditions, reference)()








