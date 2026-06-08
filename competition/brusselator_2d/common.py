from __future__ import annotations

from math import pi

import numpy as np


_GRID_SIZE = 128
_DOMAIN_LENGTH = 1.0
_DX = _DOMAIN_LENGTH / _GRID_SIZE

PROBLEM_PARAMETERS = {
    "name": "Brusselator2D",
    "grid_size": _GRID_SIZE,
    "domain_length": _DOMAIN_LENGTH,
    "t0": 0.0,
    "t1": 0.5,
    "alpha": 0.02,
    "a": 1.0,
    "b": 3.4,
    "dx": _DX,
    "inv_dx2": 1.0 / (_DX * _DX),
}

TOLERANCE_PARAMETERS = {
    "rtol": 1.0e-6,
    "atol": 1.0e-8,
    "initial_step": 1.0e-3,
}

REFERENCE_TOLERANCE_PARAMETERS = {
    "rtol": 1.0e-9,
    "atol": 1.0e-11,
}

BENCHMARK_PARAMETERS = {
    "repeats": 5,
}

_POINTS = np.linspace(
    0.0,
    PROBLEM_PARAMETERS["domain_length"],
    PROBLEM_PARAMETERS["grid_size"],
    endpoint=False,
    dtype=np.float64,
)
_XX, _YY = np.meshgrid(_POINTS, _POINTS, indexing="ij")

INITIAL_U = (
    PROBLEM_PARAMETERS["a"]
    + 0.1 * np.sin(2.0 * pi * _XX) * np.sin(2.0 * pi * _YY)
).astype(np.float64)
INITIAL_V = (
    (PROBLEM_PARAMETERS["b"] / PROBLEM_PARAMETERS["a"])
    + 0.1 * np.cos(2.0 * pi * _XX) * np.cos(2.0 * pi * _YY)
).astype(np.float64)

INITIAL_CONDITIONS = {
    "u": INITIAL_U,
    "v": INITIAL_V,
}








