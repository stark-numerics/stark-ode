from __future__ import annotations

import numpy as np

try:
    from scipy.integrate import solve_ivp
    from scipy.sparse import diags
except ImportError:  # pragma: no cover - optional dependency for local use
    SCIPY_AVAILABLE = False
else:
    SCIPY_AVAILABLE = True


def _dx(problem_parameters) -> float:
    return problem_parameters["length"] / problem_parameters["grid_size"]


def allen_cahn_rhs(problem_parameters, _t, u):
    diffusivity = problem_parameters["diffusivity"]
    inv_dx2 = 1.0 / (_dx(problem_parameters) ** 2)
    laplacian = (np.roll(u, 1) - 2.0 * u + np.roll(u, -1)) * inv_dx2
    return diffusivity * laplacian + u - u * u * u


def allen_cahn_jacobian(problem_parameters, _t, u):
    size = problem_parameters["grid_size"]
    diffusivity = problem_parameters["diffusivity"]
    inv_dx2 = 1.0 / (_dx(problem_parameters) ** 2)
    off = diffusivity * inv_dx2
    main = (-2.0 * off + 1.0 - 3.0 * u * u).astype(np.float64)
    matrix = diags(
        diagonals=[off * np.ones(size - 1), main, off * np.ones(size - 1)],
        offsets=[-1, 0, 1],
        shape=(size, size),
        format="lil",
    )
    matrix[0, size - 1] = off
    matrix[size - 1, 0] = off
    return matrix.tocsc()


def run_reference(problem_parameters, reference_tolerances, initial_conditions):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is required to generate the Allen-Cahn reference solution.")

    solution = solve_ivp(
        lambda t, y: allen_cahn_rhs(problem_parameters, t, y),
        (problem_parameters["t0"], problem_parameters["t1"]),
        initial_conditions["u"],
        method="Radau",
        jac=lambda t, y: allen_cahn_jacobian(problem_parameters, t, y),
        rtol=reference_tolerances["rtol"],
        atol=reference_tolerances["atol"],
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    return {
        "u": solution.y[:, -1].copy(),
        "steps": len(solution.t) - 1,
    }


def _prepare(method_name: str, problem_parameters, tolerance_parameters, initial_conditions, reference):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is not installed.")

    initial = initial_conditions["u"]

    def solve_once():
        solution = solve_ivp(
            lambda t, y: allen_cahn_rhs(problem_parameters, t, y),
            (problem_parameters["t0"], problem_parameters["t1"]),
            initial,
            method=method_name,
            jac=lambda t, y: allen_cahn_jacobian(problem_parameters, t, y),
            rtol=tolerance_parameters["rtol"],
            atol=tolerance_parameters["atol"],
        )
        if not solution.success:
            raise RuntimeError(solution.message)

        final = solution.y[:, -1]
        difference = final - reference["u"]
        error = np.sqrt(np.dot(difference, difference) / final.size)
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
