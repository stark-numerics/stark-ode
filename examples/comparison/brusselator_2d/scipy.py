from __future__ import annotations

import numpy as np

try:
    from scipy.integrate import solve_ivp
except ImportError:  # pragma: no cover - optional dependency for local use
    SCIPY_AVAILABLE = False
else:
    SCIPY_AVAILABLE = True


def brusselator_rhs(t, y, problem_parameters):
    del t

    grid_size = problem_parameters["grid_size"]
    pivot = grid_size * grid_size
    u = y[:pivot].reshape(grid_size, grid_size)
    v = y[pivot:].reshape(grid_size, grid_size)

    lap_u = (
        np.roll(u, 1, axis=0)
        + np.roll(u, -1, axis=0)
        + np.roll(u, 1, axis=1)
        + np.roll(u, -1, axis=1)
        - 4.0 * u
    ) * problem_parameters["inv_dx2"]
    lap_v = (
        np.roll(v, 1, axis=0)
        + np.roll(v, -1, axis=0)
        + np.roll(v, 1, axis=1)
        + np.roll(v, -1, axis=1)
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
    return np.concatenate((du.ravel(), dv.ravel()))


def run_reference(problem_parameters, reference_tolerances, initial_conditions):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is required to generate the Brusselator reference solution.")

    grid_size = problem_parameters["grid_size"]
    pivot = grid_size * grid_size
    y0 = np.concatenate((initial_conditions["u"].ravel(), initial_conditions["v"].ravel()))
    solution = solve_ivp(
        brusselator_rhs,
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
        "u": final[:pivot].reshape(grid_size, grid_size).copy(),
        "v": final[pivot:].reshape(grid_size, grid_size).copy(),
        "steps": len(solution.t) - 1,
    }


def prepare_rk45(problem_parameters, tolerance_parameters, initial_conditions, reference):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is not installed.")

    grid_size = problem_parameters["grid_size"]
    pivot = grid_size * grid_size
    y0 = np.concatenate((initial_conditions["u"].ravel(), initial_conditions["v"].ravel()))

    def solve_once():
        solution = solve_ivp(
            brusselator_rhs,
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
        u = final[:pivot].reshape(grid_size, grid_size)
        v = final[pivot:].reshape(grid_size, grid_size)
        du = u - reference["u"]
        dv = v - reference["v"]
        error = np.sqrt((np.dot(du.ravel(), du.ravel()) + np.dot(dv.ravel(), dv.ravel())) / u.size)
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

    grid_size = problem_parameters["grid_size"]
    pivot = grid_size * grid_size
    y0 = np.concatenate((initial_conditions["u"].ravel(), initial_conditions["v"].ravel()))

    def solve_once():
        solution = solve_ivp(
            brusselator_rhs,
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
        u = final[:pivot].reshape(grid_size, grid_size)
        v = final[pivot:].reshape(grid_size, grid_size)
        du = u - reference["u"]
        dv = v - reference["v"]
        error = np.sqrt((np.dot(du.ravel(), du.ravel()) + np.dot(dv.ravel(), dv.ravel())) / u.size)
        return {
            "library": "SciPy",
            "solver": "DOP853",
            "error": float(error),
            "steps": len(solution.t) - 1,
        }

    return solve_once


def run_dop853(problem_parameters, tolerance_parameters, initial_conditions, reference):
    return prepare_dop853(problem_parameters, tolerance_parameters, initial_conditions, reference)()








