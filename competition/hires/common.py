from __future__ import annotations

import numpy as np


PROBLEM_PARAMETERS = {
    "name": "HIRES",
    "t0": 0.0,
    "t1": 321.8122,
}

TOLERANCE_PARAMETERS = {
    "rtol": 1.0e-6,
    "atol": 1.0e-8,
}

REFERENCE_TOLERANCE_PARAMETERS = {
    "rtol": 1.0e-11,
    "atol": 1.0e-13,
}

STARK_PARAMETERS = {
    "step": 1.0e-3,
    "tolerance_rtol": 1.0e-6,
    "tolerance_atol": 1.0e-8,
    "resolution_atol": 1.0e-8,
    "resolution_rtol": 1.0e-8,
    "resolution_max_iterations": 32,
    "inversion_atol": 1.0e-10,
    "inversion_rtol": 1.0e-10,
    "inversion_max_iterations": 16,
}

DIFFRAX_PARAMETERS = {
    "dt0": 1.0e-3,
}

BENCHMARK_PARAMETERS = {
    "repeats": 5,
}

INITIAL_CONDITIONS = {
    "y": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057], dtype=np.float64),
}
