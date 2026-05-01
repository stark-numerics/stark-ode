from __future__ import annotations

import numpy as np


_CHAIN_SIZE = 512

PROBLEM_PARAMETERS = {
    "name": "FPUTBeta",
    "chain_size": _CHAIN_SIZE,
    "t0": 0.0,
    "t1": 400.0,
    "beta": 0.25,
    "amplitude": 0.1,
}

TOLERANCE_PARAMETERS = {
    "rtol": 1.0e-6,
    "atol": 1.0e-8,
    "initial_step": 1.0e-2,
}

REFERENCE_TOLERANCE_PARAMETERS = {
    "rtol": 1.0e-10,
    "atol": 1.0e-12,
}

BENCHMARK_PARAMETERS = {
    "repeats": 5,
}

_INDICES = np.arange(1, _CHAIN_SIZE + 1, dtype=np.float64)

INITIAL_Q = (
    PROBLEM_PARAMETERS["amplitude"] * np.sin(np.pi * _INDICES / (_CHAIN_SIZE + 1))
).astype(np.float64)
INITIAL_P = np.zeros(_CHAIN_SIZE, dtype=np.float64)

INITIAL_CONDITIONS = {
    "q": INITIAL_Q,
    "p": INITIAL_P,
}








