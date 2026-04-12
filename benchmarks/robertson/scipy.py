from __future__ import annotations

import numpy as np

try:
    from scipy.integrate import solve_ivp
except ImportError:  # pragma: no cover - optional dependency for local use
    SCIPY_AVAILABLE = False
else:
    SCIPY_AVAILABLE = True


def robertson_rhs(t, y):
    del t
    y1, y2, y3 = y
    return np.array(
        [
            -0.04 * y1 + 1.0e4 * y2 * y3,
            0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2,
            3.0e7 * y2 * y2,
        ],
        dtype=np.float64,
    )


def run_reference(problem_parameters, reference_tolerances, initial_conditions):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is required to generate the Robertson reference solution.")

    solution = solve_ivp(
        robertson_rhs,
        (problem_parameters["t0"], problem_parameters["t1"]),
        initial_conditions["y"],
        method="Radau",
        rtol=reference_tolerances["rtol"],
        atol=reference_tolerances["atol"],
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    return {
        "y": solution.y[:, -1].copy(),
        "steps": len(solution.t) - 1,
    }


def prepare_radau(problem_parameters, tolerance_parameters, initial_conditions, reference):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is not installed.")

    y0 = initial_conditions["y"]

    def solve_once():
        solution = solve_ivp(
            robertson_rhs,
            (problem_parameters["t0"], problem_parameters["t1"]),
            y0,
            method="Radau",
            rtol=tolerance_parameters["rtol"],
            atol=tolerance_parameters["atol"],
        )
        if not solution.success:
            raise RuntimeError(solution.message)

        final = solution.y[:, -1]
        error = np.sqrt(np.dot(final - reference["y"], final - reference["y"]) / final.size)
        return {
            "library": "SciPy",
            "solver": "Radau",
            "error": float(error),
            "steps": len(solution.t) - 1,
        }

    return solve_once


def run_radau(problem_parameters, tolerance_parameters, initial_conditions, reference):
    return prepare_radau(problem_parameters, tolerance_parameters, initial_conditions, reference)()


def prepare_bdf(problem_parameters, tolerance_parameters, initial_conditions, reference):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is not installed.")

    y0 = initial_conditions["y"]

    def solve_once():
        solution = solve_ivp(
            robertson_rhs,
            (problem_parameters["t0"], problem_parameters["t1"]),
            y0,
            method="BDF",
            rtol=tolerance_parameters["rtol"],
            atol=tolerance_parameters["atol"],
        )
        if not solution.success:
            raise RuntimeError(solution.message)

        final = solution.y[:, -1]
        error = np.sqrt(np.dot(final - reference["y"], final - reference["y"]) / final.size)
        return {
            "library": "SciPy",
            "solver": "BDF",
            "error": float(error),
            "steps": len(solution.t) - 1,
        }

    return solve_once


def run_bdf(problem_parameters, tolerance_parameters, initial_conditions, reference):
    return prepare_bdf(problem_parameters, tolerance_parameters, initial_conditions, reference)()
