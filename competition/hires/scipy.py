from __future__ import annotations

import numpy as np

try:
    from scipy.integrate import solve_ivp
except ImportError:  # pragma: no cover - optional dependency for local use
    SCIPY_AVAILABLE = False
else:
    SCIPY_AVAILABLE = True


def hires_rhs(t, y):
    del t
    y0, y1, y2, y3, y4, y5, y6, y7 = y
    reaction = 280.0 * y5 * y7
    return np.array(
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
        dtype=np.float64,
    )


def run_reference(problem_parameters, reference_tolerances, initial_conditions):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is required to generate the HIRES reference solution.")

    solution = solve_ivp(
        hires_rhs,
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


def _prepare(method_name: str, problem_parameters, tolerance_parameters, initial_conditions, reference):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is not installed.")

    y0 = initial_conditions["y"]

    def solve_once():
        solution = solve_ivp(
            hires_rhs,
            (problem_parameters["t0"], problem_parameters["t1"]),
            y0,
            method=method_name,
            rtol=tolerance_parameters["rtol"],
            atol=tolerance_parameters["atol"],
        )
        if not solution.success:
            raise RuntimeError(solution.message)

        final = solution.y[:, -1]
        error = np.sqrt(np.dot(final - reference["y"], final - reference["y"]) / final.size)
        return {
            "library": "SciPy",
            "solver": method_name,
            "error": float(error),
            "steps": len(solution.t) - 1,
        }

    return solve_once


def prepare_radau(problem_parameters, tolerance_parameters, initial_conditions, reference):
    return _prepare("Radau", problem_parameters, tolerance_parameters, initial_conditions, reference)


def prepare_bdf(problem_parameters, tolerance_parameters, initial_conditions, reference):
    return _prepare("BDF", problem_parameters, tolerance_parameters, initial_conditions, reference)
