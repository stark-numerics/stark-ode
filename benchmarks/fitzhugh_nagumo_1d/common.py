from __future__ import annotations

import numpy as np


PROBLEM_PARAMETERS = {
    "name": "FitzHugh-Nagumo 1D",
    "grid_size": 128,
    "length": 40.0,
    "diffusivity_u": 1.0,
    "epsilon": 0.08,
    "a": 0.7,
    "b": 0.8,
    "t0": 0.0,
    "t1": 18.0,
}
PROBLEM_PARAMETERS["dx"] = PROBLEM_PARAMETERS["length"] / PROBLEM_PARAMETERS["grid_size"]
PROBLEM_PARAMETERS["inv_dx2"] = 1.0 / (PROBLEM_PARAMETERS["dx"] * PROBLEM_PARAMETERS["dx"])

TOLERANCE_PARAMETERS = {
    "rtol": 1.0e-5,
    "atol": 1.0e-6,
}

REFERENCE_TOLERANCE_PARAMETERS = {
    "rtol": 1.0e-9,
    "atol": 1.0e-11,
}

STARK_PARAMETERS = {
    "step": 5.0e-3,
    "tolerance_rtol": 1.0e-5,
    "tolerance_atol": 1.0e-6,
    "resolution_atol": 1.0e-7,
    "resolution_rtol": 1.0e-7,
    "resolution_max_iterations": 24,
    "inversion_atol": 1.0e-7,
    "inversion_rtol": 1.0e-7,
    "inversion_max_iterations": 24,
    "inversion_restart": 12,
}

DIFFRAX_PARAMETERS = {
    "dt0": 5.0e-3,
}

BENCHMARK_PARAMETERS = {
    "repeats": 5,
    "checkpoints": 36,
}

_x = np.linspace(0.0, PROBLEM_PARAMETERS["length"], PROBLEM_PARAMETERS["grid_size"], endpoint=False)
_u = -1.2 + 2.4 * np.exp(-((_x - 0.3 * PROBLEM_PARAMETERS["length"]) ** 2) / 1.5)
_v = -0.62 + 0.1 * np.exp(-((_x - 0.3 * PROBLEM_PARAMETERS["length"]) ** 2) / 1.5)

INITIAL_CONDITIONS = {
    "x": _x,
    "u": _u.astype(np.float64),
    "v": _v.astype(np.float64),
}
