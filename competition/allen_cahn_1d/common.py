from __future__ import annotations

import numpy as np


PROBLEM_PARAMETERS = {
    "name": "Allen-Cahn 1D",
    "t0": 0.0,
    "t1": 0.05,
    "length": 6.0,
    "grid_size": 96,
    "diffusivity": 0.08,
}

TOLERANCE_PARAMETERS = {
    "rtol": 1.0e-6,
    "atol": 1.0e-8,
}

REFERENCE_TOLERANCE_PARAMETERS = {
    "rtol": 1.0e-9,
    "atol": 1.0e-11,
}

STARK_PARAMETERS = {
    "step": 1.0e-3,
    "tolerance_rtol": 1.0e-5,
    "tolerance_atol": 1.0e-7,
    "resolution_atol": 1.0e-7,
    "resolution_rtol": 1.0e-7,
    "resolution_max_iterations": 10,
    "inversion_atol": 1.0e-8,
    "inversion_rtol": 1.0e-8,
    "inversion_max_iterations": 24,
    "inversion_restart": 12,
}

DIFFRAX_PARAMETERS = {
    "dt0": 1.0e-3,
}

BENCHMARK_PARAMETERS = {
    "repeats": 3,
}


def grid(problem_parameters=PROBLEM_PARAMETERS) -> np.ndarray:
    return np.linspace(
        0.0,
        problem_parameters["length"],
        problem_parameters["grid_size"],
        endpoint=False,
        dtype=np.float64,
    )


def initial_profile(problem_parameters=PROBLEM_PARAMETERS) -> np.ndarray:
    x = grid(problem_parameters)
    k = 2.0 * np.pi / problem_parameters["length"]
    return (0.5 * np.sin(k * x) + 0.5 * np.sin(3.0 * k * x)).astype(np.float64)


INITIAL_CONDITIONS = {
    "u": initial_profile(),
}
