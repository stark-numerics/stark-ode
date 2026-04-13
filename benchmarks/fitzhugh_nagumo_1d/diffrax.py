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


def _split_state(y, grid_size: int):
    return y[:grid_size], y[grid_size:]


def _laplacian_periodic(field, inv_dx2: float):
    return (jnp.roll(field, 1) - 2.0 * field + jnp.roll(field, -1)) * inv_dx2


def vector_field(t, y, problem_parameters):
    del t
    grid_size = problem_parameters["grid_size"]
    diffusivity_u = problem_parameters["diffusivity_u"]
    epsilon = problem_parameters["epsilon"]
    a = problem_parameters["a"]
    b = problem_parameters["b"]
    inv_dx2 = problem_parameters["inv_dx2"]

    u, v = _split_state(y, grid_size)
    du = diffusivity_u * _laplacian_periodic(u, inv_dx2) + u - (u * u * u) / 3.0 - v
    dv = epsilon * (u + a - b * v)
    return jnp.concatenate((du, dv))


def _error(final: np.ndarray, reference: dict[str, np.ndarray], grid_size: int) -> float:
    u = final[:grid_size]
    v = final[grid_size:]
    du = u - reference["u"]
    dv = v - reference["v"]
    return float(np.sqrt((np.dot(du, du) + np.dot(dv, dv)) / (du.size + dv.size)))


def prepare_kvaerno5(problem_parameters, tolerance_parameters, diffrax_parameters, initial_conditions, reference):
    if not DIFFRAX_AVAILABLE:
        raise RuntimeError("Diffrax/JAX is not installed.")

    term = dfx.ODETerm(vector_field)
    solver = dfx.Kvaerno5()
    saveat = dfx.SaveAt(t1=True)
    controller = dfx.PIDController(rtol=tolerance_parameters["rtol"], atol=tolerance_parameters["atol"])
    y0 = jnp.asarray(np.concatenate((initial_conditions["u"], initial_conditions["v"])))

    @eqx.filter_jit
    def solve(y0):
        return dfx.diffeqsolve(
            term,
            solver,
            t0=problem_parameters["t0"],
            t1=problem_parameters["t1"],
            dt0=diffrax_parameters["dt0"],
            y0=y0,
            args=problem_parameters,
            saveat=saveat,
            stepsize_controller=controller,
            max_steps=1_000_000,
        )

    def solve_once():
        solution = solve(y0)
        jax.block_until_ready(solution.ys)
        final = np.asarray(solution.ys[0])
        return {
            "library": "Diffrax",
            "solver": "Kvaerno5",
            "error": _error(final, reference, problem_parameters["grid_size"]),
            "steps": int(solution.stats["num_steps"]),
        }

    return solve_once
