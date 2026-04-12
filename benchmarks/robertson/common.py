from __future__ import annotations

import numpy as np


PROBLEM_PARAMETERS = {
    "name": "Robertson",
    "t0": 0.0,
    "t1": 1.0,
}

TOLERANCE_PARAMETERS = {
    "rtol": 1.0e-6,
    "atol": 1.0e-10,
}

REFERENCE_TOLERANCE_PARAMETERS = {
    "rtol": 1.0e-12,
    "atol": 1.0e-14,
}

STARK_PARAMETERS = {
    "step": 1.0e-4,
    "tolerance_rtol": 1.0e-6,
    "tolerance_atol": 1.0e-10,
    "resolution_atol": 1.0e-8,
    "resolution_rtol": 1.0e-8,
    "resolution_max_iterations": 32,
    "inversion_atol": 1.0e-8,
    "inversion_rtol": 1.0e-8,
    "inversion_max_iterations": 16,
    "inversion_restart": 8,
}

DIFFRAX_PARAMETERS = {
    "dt0": 2.0e-4,
}

BENCHMARK_PARAMETERS = {
    "repeats": 5,
}

INITIAL_CONDITIONS = {
    "y": np.array([1.0, 0.0, 0.0], dtype=np.float64),
}
