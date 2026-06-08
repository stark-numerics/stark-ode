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

    problem_parameters = args
    u, v = y
    lap_u = (
        jnp.roll(u, 1, axis=0)
        + jnp.roll(u, -1, axis=0)
        + jnp.roll(u, 1, axis=1)
        + jnp.roll(u, -1, axis=1)
        - 4.0 * u
    ) * problem_parameters["inv_dx2"]
    lap_v = (
        jnp.roll(v, 1, axis=0)
        + jnp.roll(v, -1, axis=0)
        + jnp.roll(v, 1, axis=1)
        + jnp.roll(v, -1, axis=1)
        - 4.0 * v
    ) * problem_parameters["inv_dx2"]
    reaction = u * u * v
    du = (
        problem_parameters["alpha"] * lap_u
        + problem_parameters["a"]
        + reaction
        - (problem_parameters["b"] + 1.0) * u
    )
    dv = problem_parameters["alpha"] * lap_v + problem_parameters["b"] * u - reaction
    return du, dv


def prepare_tsit5(problem_parameters, tolerance_parameters, initial_conditions, reference):
    if not DIFFRAX_AVAILABLE:
        raise RuntimeError("Diffrax/JAX is not installed.")

    term = dfx.ODETerm(vector_field)
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(t1=True)
    stepsize_controller = dfx.PIDController(
        rtol=tolerance_parameters["rtol"],
        atol=tolerance_parameters["atol"],
    )
    args = {
        "alpha": problem_parameters["alpha"],
        "a": problem_parameters["a"],
        "b": problem_parameters["b"],
        "inv_dx2": problem_parameters["inv_dx2"],
    }
    y0 = (jnp.asarray(initial_conditions["u"]), jnp.asarray(initial_conditions["v"]))

    @eqx.filter_jit
    def solve(y0):
        return dfx.diffeqsolve(
            term,
            solver,
            t0=problem_parameters["t0"],
            t1=problem_parameters["t1"],
            dt0=None,
            y0=y0,
            args=args,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=1_000_000,
        )

    def solve_once():
        solution = solve(y0)
        jax.block_until_ready(solution.ys)
        final_u, final_v = solution.ys
        u = np.asarray(final_u[0])
        v = np.asarray(final_v[0])
        du = u - reference["u"]
        dv = v - reference["v"]
        error = np.sqrt((np.dot(du.ravel(), du.ravel()) + np.dot(dv.ravel(), dv.ravel())) / u.size)
        return {
            "library": "Diffrax",
            "solver": "Tsit5",
            "error": float(error),
            "steps": int(solution.stats["num_steps"]),
        }

    return solve_once


def run_tsit5(problem_parameters, tolerance_parameters, initial_conditions, reference):
    return prepare_tsit5(problem_parameters, tolerance_parameters, initial_conditions, reference)()


def prepare_dopri5(problem_parameters, tolerance_parameters, initial_conditions, reference):
    if not DIFFRAX_AVAILABLE:
        raise RuntimeError("Diffrax/JAX is not installed.")

    term = dfx.ODETerm(vector_field)
    solver = dfx.Dopri5()
    saveat = dfx.SaveAt(t1=True)
    stepsize_controller = dfx.PIDController(
        rtol=tolerance_parameters["rtol"],
        atol=tolerance_parameters["atol"],
    )
    args = {
        "alpha": problem_parameters["alpha"],
        "a": problem_parameters["a"],
        "b": problem_parameters["b"],
        "inv_dx2": problem_parameters["inv_dx2"],
    }
    y0 = (jnp.asarray(initial_conditions["u"]), jnp.asarray(initial_conditions["v"]))

    @eqx.filter_jit
    def solve(y0):
        return dfx.diffeqsolve(
            term,
            solver,
            t0=problem_parameters["t0"],
            t1=problem_parameters["t1"],
            dt0=None,
            y0=y0,
            args=args,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=1_000_000,
        )

    def solve_once():
        solution = solve(y0)
        jax.block_until_ready(solution.ys)
        final_u, final_v = solution.ys
        u = np.asarray(final_u[0])
        v = np.asarray(final_v[0])
        du = u - reference["u"]
        dv = v - reference["v"]
        error = np.sqrt((np.dot(du.ravel(), du.ravel()) + np.dot(dv.ravel(), dv.ravel())) / u.size)
        return {
            "library": "Diffrax",
            "solver": "Dopri5",
            "error": float(error),
            "steps": int(solution.stats["num_steps"]),
        }

    return solve_once


def run_dopri5(problem_parameters, tolerance_parameters, initial_conditions, reference):
    return prepare_dopri5(problem_parameters, tolerance_parameters, initial_conditions, reference)()








