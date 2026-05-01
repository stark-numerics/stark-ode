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

    beta = args["beta"]
    q, p = y
    left = jnp.concatenate((jnp.zeros((1,), dtype=q.dtype), q[:-1]))
    right = jnp.concatenate((q[1:], jnp.zeros((1,), dtype=q.dtype)))
    dq = p
    dp = right - 2.0 * q + left + beta * ((right - q) ** 3 - (q - left) ** 3)
    return dq, dp


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
        "beta": problem_parameters["beta"],
    }
    y0 = (jnp.asarray(initial_conditions["q"]), jnp.asarray(initial_conditions["p"]))

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
        final_q, final_p = solution.ys
        q = np.asarray(final_q[0])
        p = np.asarray(final_p[0])
        dq = q - reference["q"]
        dp = p - reference["p"]
        error = np.sqrt((np.dot(dq.ravel(), dq.ravel()) + np.dot(dp.ravel(), dp.ravel())) / q.size)
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
        "beta": problem_parameters["beta"],
    }
    y0 = (jnp.asarray(initial_conditions["q"]), jnp.asarray(initial_conditions["p"]))

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
        final_q, final_p = solution.ys
        q = np.asarray(final_q[0])
        p = np.asarray(final_p[0])
        dq = q - reference["q"]
        dp = p - reference["p"]
        error = np.sqrt((np.dot(dq.ravel(), dq.ravel()) + np.dot(dp.ravel(), dp.ravel())) / q.size)
        return {
            "library": "Diffrax",
            "solver": "Dopri5",
            "error": float(error),
            "steps": int(solution.stats["num_steps"]),
        }

    return solve_once


def run_dopri5(problem_parameters, tolerance_parameters, initial_conditions, reference):
    return prepare_dopri5(problem_parameters, tolerance_parameters, initial_conditions, reference)()








