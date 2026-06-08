from __future__ import annotations

import numpy as np

try:
    from scipy.integrate import solve_ivp
except ImportError:  # pragma: no cover - optional dependency for local use
    SCIPY_AVAILABLE = False
else:
    SCIPY_AVAILABLE = True


def fput_rhs(t, y, problem_parameters):
    del t

    pivot = problem_parameters["chain_size"]
    beta = problem_parameters["beta"]
    q = y[:pivot]
    p = y[pivot:]

    left = np.empty_like(q)
    right = np.empty_like(q)
    left[0] = 0.0
    left[1:] = q[:-1]
    right[-1] = 0.0
    right[:-1] = q[1:]

    dq = p
    dp = right - 2.0 * q + left + beta * ((right - q) ** 3 - (q - left) ** 3)
    return np.concatenate((dq, dp))


def run_reference(problem_parameters, reference_tolerances, initial_conditions):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is required to generate the FPUT reference solution.")

    pivot = problem_parameters["chain_size"]
    y0 = np.concatenate((initial_conditions["q"], initial_conditions["p"]))
    solution = solve_ivp(
        fput_rhs,
        (problem_parameters["t0"], problem_parameters["t1"]),
        y0,
        args=(problem_parameters,),
        method="DOP853",
        rtol=reference_tolerances["rtol"],
        atol=reference_tolerances["atol"],
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    final = solution.y[:, -1]
    return {
        "q": final[:pivot].copy(),
        "p": final[pivot:].copy(),
        "steps": len(solution.t) - 1,
    }


def prepare_rk45(problem_parameters, tolerance_parameters, initial_conditions, reference):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is not installed.")

    pivot = problem_parameters["chain_size"]
    y0 = np.concatenate((initial_conditions["q"], initial_conditions["p"]))

    def solve_once():
        solution = solve_ivp(
            fput_rhs,
            (problem_parameters["t0"], problem_parameters["t1"]),
            y0,
            args=(problem_parameters,),
            method="RK45",
            rtol=tolerance_parameters["rtol"],
            atol=tolerance_parameters["atol"],
        )
        if not solution.success:
            raise RuntimeError(solution.message)

        final = solution.y[:, -1]
        q = final[:pivot]
        p = final[pivot:]
        dq = q - reference["q"]
        dp = p - reference["p"]
        error = np.sqrt((np.dot(dq.ravel(), dq.ravel()) + np.dot(dp.ravel(), dp.ravel())) / q.size)
        return {
            "library": "SciPy",
            "solver": "RK45",
            "error": float(error),
            "steps": len(solution.t) - 1,
        }

    return solve_once


def run_rk45(problem_parameters, tolerance_parameters, initial_conditions, reference):
    return prepare_rk45(problem_parameters, tolerance_parameters, initial_conditions, reference)()


def prepare_dop853(problem_parameters, tolerance_parameters, initial_conditions, reference):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is not installed.")

    pivot = problem_parameters["chain_size"]
    y0 = np.concatenate((initial_conditions["q"], initial_conditions["p"]))

    def solve_once():
        solution = solve_ivp(
            fput_rhs,
            (problem_parameters["t0"], problem_parameters["t1"]),
            y0,
            args=(problem_parameters,),
            method="DOP853",
            rtol=tolerance_parameters["rtol"],
            atol=tolerance_parameters["atol"],
        )
        if not solution.success:
            raise RuntimeError(solution.message)

        final = solution.y[:, -1]
        q = final[:pivot]
        p = final[pivot:]
        dq = q - reference["q"]
        dp = p - reference["p"]
        error = np.sqrt((np.dot(dq.ravel(), dq.ravel()) + np.dot(dp.ravel(), dp.ravel())) / q.size)
        return {
            "library": "SciPy",
            "solver": "DOP853",
            "error": float(error),
            "steps": len(solution.t) - 1,
        }

    return solve_once


def run_dop853(problem_parameters, tolerance_parameters, initial_conditions, reference):
    return prepare_dop853(problem_parameters, tolerance_parameters, initial_conditions, reference)()








