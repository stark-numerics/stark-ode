from __future__ import annotations

import numpy as np

try:
    from scipy.integrate import solve_ivp
except ImportError:  # pragma: no cover - optional dependency for local use
    SCIPY_AVAILABLE = False
else:
    SCIPY_AVAILABLE = True


def _split_state(y: np.ndarray, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    return y[:grid_size], y[grid_size:]


def _laplacian_periodic(field: np.ndarray, inv_dx2: float) -> np.ndarray:
    return (np.roll(field, 1) - 2.0 * field + np.roll(field, -1)) * inv_dx2


def fitzhugh_nagumo_rhs(t, y, problem_parameters):
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
    return np.concatenate((du, dv))


def _error(final: np.ndarray, reference: dict[str, np.ndarray], grid_size: int) -> float:
    u, v = _split_state(final, grid_size)
    du = u - reference["u"]
    dv = v - reference["v"]
    return float(np.sqrt((np.dot(du, du) + np.dot(dv, dv)) / (du.size + dv.size)))


def run_reference(problem_parameters, reference_tolerances, initial_conditions):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is required to generate the FitzHugh-Nagumo reference solution.")

    y0 = np.concatenate((initial_conditions["u"], initial_conditions["v"]))
    solution = solve_ivp(
        fitzhugh_nagumo_rhs,
        (problem_parameters["t0"], problem_parameters["t1"]),
        y0,
        method="Radau",
        args=(problem_parameters,),
        rtol=reference_tolerances["rtol"],
        atol=reference_tolerances["atol"],
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    u, v = _split_state(solution.y[:, -1], problem_parameters["grid_size"])
    return {
        "u": u.copy(),
        "v": v.copy(),
        "steps": len(solution.t) - 1,
    }


def prepare_radau(problem_parameters, tolerance_parameters, initial_conditions, reference):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is not installed.")

    y0 = np.concatenate((initial_conditions["u"], initial_conditions["v"]))

    def solve_once():
        solution = solve_ivp(
            fitzhugh_nagumo_rhs,
            (problem_parameters["t0"], problem_parameters["t1"]),
            y0,
            method="Radau",
            args=(problem_parameters,),
            rtol=tolerance_parameters["rtol"],
            atol=tolerance_parameters["atol"],
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        return {
            "library": "SciPy",
            "solver": "Radau",
            "error": _error(solution.y[:, -1], reference, problem_parameters["grid_size"]),
            "steps": len(solution.t) - 1,
        }

    return solve_once


def prepare_bdf(problem_parameters, tolerance_parameters, initial_conditions, reference):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is not installed.")

    y0 = np.concatenate((initial_conditions["u"], initial_conditions["v"]))

    def solve_once():
        solution = solve_ivp(
            fitzhugh_nagumo_rhs,
            (problem_parameters["t0"], problem_parameters["t1"]),
            y0,
            method="BDF",
            args=(problem_parameters,),
            rtol=tolerance_parameters["rtol"],
            atol=tolerance_parameters["atol"],
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        return {
            "library": "SciPy",
            "solver": "BDF",
            "error": _error(solution.y[:, -1], reference, problem_parameters["grid_size"]),
            "steps": len(solution.t) - 1,
        }

    return solve_once








