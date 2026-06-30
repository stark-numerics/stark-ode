from __future__ import annotations

from typing import Any

import numpy as np

from competition.allen_cahn_1d import common

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


def vector_field(t, u, args):
    del t
    diffusivity, inv_dx2 = args
    laplacian = (jnp.roll(u, 1) - 2.0 * u + jnp.roll(u, -1)) * inv_dx2
    return diffusivity * laplacian + u - u * u * u


def kvaerno5_solver(problem_parameters, tolerance_parameters, diffrax_parameters, initial_conditions, reference):
    if not DIFFRAX_AVAILABLE:
        raise RuntimeError("Diffrax/JAX is not installed.")

    dx = problem_parameters["length"] / problem_parameters["grid_size"]
    args = (problem_parameters["diffusivity"], 1.0 / (dx * dx))
    term = dfx.ODETerm(vector_field)
    solver = dfx.Kvaerno5()
    saveat = dfx.SaveAt(t1=True)
    controller = dfx.PIDController(
        rtol=tolerance_parameters["rtol"],
        atol=tolerance_parameters["atol"],
    )
    y0 = jnp.asarray(initial_conditions["u"], dtype=jnp.float64)

    @eqx.filter_jit
    def solve(y0):
        return dfx.diffeqsolve(
            term,
            solver,
            t0=problem_parameters["t0"],
            t1=problem_parameters["t1"],
            dt0=diffrax_parameters["dt0"],
            y0=y0,
            args=args,
            saveat=saveat,
            stepsize_controller=controller,
            max_steps=1_000_000,
        )

    def solve_once():
        solution = solve(y0)
        jax.block_until_ready(solution.ys)
        final = np.asarray(solution.ys[0])
        difference = final - reference["u"]
        error = np.sqrt(np.dot(difference, difference) / final.size)
        return {
            "library": "Diffrax",
            "solver": "Kvaerno5",
            "error": float(error),
            "steps": int(solution.stats["num_steps"]),
        }

    return solve_once


def prepare_kvaerno5(problem_parameters, tolerance_parameters, initial_conditions, reference):
    return kvaerno5_solver(
        problem_parameters,
        tolerance_parameters,
        common.DIFFRAX_PARAMETERS,
        initial_conditions,
        reference,
    )
