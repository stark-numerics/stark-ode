from __future__ import annotations

from math import sqrt

import numpy as np

from stark import Marcher, Integrator, Interval, Tolerance
from stark.scheme_library.adaptive import SchemeCashKarp, SchemeDormandPrince

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional benchmark accelerator
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True


def _optional_njit(function):
    return njit(cache=True)(function) if NUMBA_AVAILABLE else function


@_optional_njit
def _apply_kernel(origin_q, origin_p, dq, dp, result_q, result_p):
    size = origin_q.size
    for i in range(size):
        result_q[i] = origin_q[i] + dq[i]
        result_p[i] = origin_p[i] + dp[i]


@_optional_njit
def _norm_kernel(dq, dp):
    size = dq.size
    total = 0.0
    for i in range(size):
        total += dq[i] * dq[i] + dp[i] * dp[i]
    return (total / size) ** 0.5


@_optional_njit
def _scale_kernel(out_dq, out_dp, a, x_dq, x_dp):
    size = out_dq.size
    for i in range(size):
        out_dq[i] = a * x_dq[i]
        out_dp[i] = a * x_dp[i]


@_optional_njit
def _combine2_kernel(out_dq, out_dp, a0, x0_dq, x0_dp, a1, x1_dq, x1_dp):
    size = out_dq.size
    for i in range(size):
        out_dq[i] = a0 * x0_dq[i] + a1 * x1_dq[i]
        out_dp[i] = a0 * x0_dp[i] + a1 * x1_dp[i]


@_optional_njit
def _combine3_kernel(out_dq, out_dp, a0, x0_dq, x0_dp, a1, x1_dq, x1_dp, a2, x2_dq, x2_dp):
    size = out_dq.size
    for i in range(size):
        out_dq[i] = a0 * x0_dq[i] + a1 * x1_dq[i] + a2 * x2_dq[i]
        out_dp[i] = a0 * x0_dp[i] + a1 * x1_dp[i] + a2 * x2_dp[i]


@_optional_njit
def _combine4_kernel(
    out_dq,
    out_dp,
    a0,
    x0_dq,
    x0_dp,
    a1,
    x1_dq,
    x1_dp,
    a2,
    x2_dq,
    x2_dp,
    a3,
    x3_dq,
    x3_dp,
):
    size = out_dq.size
    for i in range(size):
        out_dq[i] = a0 * x0_dq[i] + a1 * x1_dq[i] + a2 * x2_dq[i] + a3 * x3_dq[i]
        out_dp[i] = a0 * x0_dp[i] + a1 * x1_dp[i] + a2 * x2_dp[i] + a3 * x3_dp[i]


@_optional_njit
def _combine5_kernel(
    out_dq,
    out_dp,
    a0,
    x0_dq,
    x0_dp,
    a1,
    x1_dq,
    x1_dp,
    a2,
    x2_dq,
    x2_dp,
    a3,
    x3_dq,
    x3_dp,
    a4,
    x4_dq,
    x4_dp,
):
    size = out_dq.size
    for i in range(size):
        out_dq[i] = a0 * x0_dq[i] + a1 * x1_dq[i] + a2 * x2_dq[i] + a3 * x3_dq[i] + a4 * x4_dq[i]
        out_dp[i] = a0 * x0_dp[i] + a1 * x1_dp[i] + a2 * x2_dp[i] + a3 * x3_dp[i] + a4 * x4_dp[i]


@_optional_njit
def _combine6_kernel(
    out_dq,
    out_dp,
    a0,
    x0_dq,
    x0_dp,
    a1,
    x1_dq,
    x1_dp,
    a2,
    x2_dq,
    x2_dp,
    a3,
    x3_dq,
    x3_dp,
    a4,
    x4_dq,
    x4_dp,
    a5,
    x5_dq,
    x5_dp,
):
    size = out_dq.size
    for i in range(size):
        out_dq[i] = (
            a0 * x0_dq[i]
            + a1 * x1_dq[i]
            + a2 * x2_dq[i]
            + a3 * x3_dq[i]
            + a4 * x4_dq[i]
            + a5 * x5_dq[i]
        )
        out_dp[i] = (
            a0 * x0_dp[i]
            + a1 * x1_dp[i]
            + a2 * x2_dp[i]
            + a3 * x3_dp[i]
            + a4 * x4_dp[i]
            + a5 * x5_dp[i]
        )


@_optional_njit
def _combine7_kernel(
    out_dq,
    out_dp,
    a0,
    x0_dq,
    x0_dp,
    a1,
    x1_dq,
    x1_dp,
    a2,
    x2_dq,
    x2_dp,
    a3,
    x3_dq,
    x3_dp,
    a4,
    x4_dq,
    x4_dp,
    a5,
    x5_dq,
    x5_dp,
    a6,
    x6_dq,
    x6_dp,
):
    size = out_dq.size
    for i in range(size):
        out_dq[i] = (
            a0 * x0_dq[i]
            + a1 * x1_dq[i]
            + a2 * x2_dq[i]
            + a3 * x3_dq[i]
            + a4 * x4_dq[i]
            + a5 * x5_dq[i]
            + a6 * x6_dq[i]
        )
        out_dp[i] = (
            a0 * x0_dp[i]
            + a1 * x1_dp[i]
            + a2 * x2_dp[i]
            + a3 * x3_dp[i]
            + a4 * x4_dp[i]
            + a5 * x5_dp[i]
            + a6 * x6_dp[i]
        )


@_optional_njit
def _rhs_kernel(q, p, dq, dp, beta):
    size = q.size
    for i in range(size):
        left = 0.0 if i == 0 else q[i - 1]
        right = 0.0 if i == size - 1 else q[i + 1]
        qi = q[i]
        dq[i] = p[i]
        dp[i] = right - 2.0 * qi + left + beta * ((right - qi) ** 3 - (qi - left) ** 3)


def _scale_translation(out, a, x):
    if NUMBA_AVAILABLE:
        _scale_kernel(out.dq, out.dp, a, x.dq, x.dp)
    else:
        np.multiply(x.dq, a, out=out.dq)
        np.multiply(x.dp, a, out=out.dp)
    return out


def _combine2_translation(out, a0, x0, a1, x1):
    if NUMBA_AVAILABLE:
        _combine2_kernel(out.dq, out.dp, a0, x0.dq, x0.dp, a1, x1.dq, x1.dp)
    else:
        np.multiply(x0.dq, a0, out=out.dq)
        out.dq += a1 * x1.dq
        np.multiply(x0.dp, a0, out=out.dp)
        out.dp += a1 * x1.dp
    return out


def _combine3_translation(out, a0, x0, a1, x1, a2, x2):
    if NUMBA_AVAILABLE:
        _combine3_kernel(out.dq, out.dp, a0, x0.dq, x0.dp, a1, x1.dq, x1.dp, a2, x2.dq, x2.dp)
    else:
        np.multiply(x0.dq, a0, out=out.dq)
        out.dq += a1 * x1.dq
        out.dq += a2 * x2.dq
        np.multiply(x0.dp, a0, out=out.dp)
        out.dp += a1 * x1.dp
        out.dp += a2 * x2.dp
    return out


def _combine4_translation(out, a0, x0, a1, x1, a2, x2, a3, x3):
    if NUMBA_AVAILABLE:
        _combine4_kernel(out.dq, out.dp, a0, x0.dq, x0.dp, a1, x1.dq, x1.dp, a2, x2.dq, x2.dp, a3, x3.dq, x3.dp)
    else:
        np.multiply(x0.dq, a0, out=out.dq)
        out.dq += a1 * x1.dq
        out.dq += a2 * x2.dq
        out.dq += a3 * x3.dq
        np.multiply(x0.dp, a0, out=out.dp)
        out.dp += a1 * x1.dp
        out.dp += a2 * x2.dp
        out.dp += a3 * x3.dp
    return out


def _combine5_translation(out, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4):
    if NUMBA_AVAILABLE:
        _combine5_kernel(
            out.dq,
            out.dp,
            a0,
            x0.dq,
            x0.dp,
            a1,
            x1.dq,
            x1.dp,
            a2,
            x2.dq,
            x2.dp,
            a3,
            x3.dq,
            x3.dp,
            a4,
            x4.dq,
            x4.dp,
        )
    else:
        np.multiply(x0.dq, a0, out=out.dq)
        out.dq += a1 * x1.dq
        out.dq += a2 * x2.dq
        out.dq += a3 * x3.dq
        out.dq += a4 * x4.dq
        np.multiply(x0.dp, a0, out=out.dp)
        out.dp += a1 * x1.dp
        out.dp += a2 * x2.dp
        out.dp += a3 * x3.dp
        out.dp += a4 * x4.dp
    return out


def _combine6_translation(out, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5):
    if NUMBA_AVAILABLE:
        _combine6_kernel(
            out.dq,
            out.dp,
            a0,
            x0.dq,
            x0.dp,
            a1,
            x1.dq,
            x1.dp,
            a2,
            x2.dq,
            x2.dp,
            a3,
            x3.dq,
            x3.dp,
            a4,
            x4.dq,
            x4.dp,
            a5,
            x5.dq,
            x5.dp,
        )
    else:
        np.multiply(x0.dq, a0, out=out.dq)
        out.dq += a1 * x1.dq
        out.dq += a2 * x2.dq
        out.dq += a3 * x3.dq
        out.dq += a4 * x4.dq
        out.dq += a5 * x5.dq
        np.multiply(x0.dp, a0, out=out.dp)
        out.dp += a1 * x1.dp
        out.dp += a2 * x2.dp
        out.dp += a3 * x3.dp
        out.dp += a4 * x4.dp
        out.dp += a5 * x5.dp
    return out


def _combine7_translation(out, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6):
    if NUMBA_AVAILABLE:
        _combine7_kernel(
            out.dq,
            out.dp,
            a0,
            x0.dq,
            x0.dp,
            a1,
            x1.dq,
            x1.dp,
            a2,
            x2.dq,
            x2.dp,
            a3,
            x3.dq,
            x3.dp,
            a4,
            x4.dq,
            x4.dp,
            a5,
            x5.dq,
            x5.dp,
            a6,
            x6.dq,
            x6.dp,
        )
    else:
        np.multiply(x0.dq, a0, out=out.dq)
        out.dq += a1 * x1.dq
        out.dq += a2 * x2.dq
        out.dq += a3 * x3.dq
        out.dq += a4 * x4.dq
        out.dq += a5 * x5.dq
        out.dq += a6 * x6.dq
        np.multiply(x0.dp, a0, out=out.dp)
        out.dp += a1 * x1.dp
        out.dp += a2 * x2.dp
        out.dp += a3 * x3.dp
        out.dp += a4 * x4.dp
        out.dp += a5 * x5.dp
        out.dp += a6 * x6.dp
    return out


class FPUTState:
    __slots__ = ("q", "p")

    def __init__(self, q, p):
        self.q = q
        self.p = p

    def __repr__(self) -> str:
        return f"FPUTState(size={self.q.size})"

    __str__ = __repr__

    def error_against(self, reference):
        dq = self.q - reference["q"]
        dp = self.p - reference["p"]
        energy = np.dot(dq.ravel(), dq.ravel()) + np.dot(dp.ravel(), dp.ravel())
        return sqrt(float(energy) / self.q.size)


class FPUTTranslation:
    __slots__ = ("dq", "dp")

    def __init__(self, dq, dp):
        self.dq = dq
        self.dp = dp

    def __repr__(self) -> str:
        return f"FPUTTranslation(size={self.dq.size}, norm={self.norm():.6g})"

    __str__ = __repr__

    def __call__(self, origin, result):
        if NUMBA_AVAILABLE:
            _apply_kernel(origin.q, origin.p, self.dq, self.dp, result.q, result.p)
        else:
            np.add(origin.q, self.dq, out=result.q)
            np.add(origin.p, self.dp, out=result.p)

    def norm(self):
        if NUMBA_AVAILABLE:
            return float(_norm_kernel(self.dq, self.dp))
        energy = np.dot(self.dq.ravel(), self.dq.ravel()) + np.dot(self.dp.ravel(), self.dp.ravel())
        return sqrt(float(energy) / self.dq.size)

    def __add__(self, other):
        return FPUTTranslation(self.dq + other.dq, self.dp + other.dp)

    def __rmul__(self, scalar):
        return FPUTTranslation(scalar * self.dq, scalar * self.dp)

    linear_combine = [
        _scale_translation,
        _combine2_translation,
        _combine3_translation,
        _combine4_translation,
        _combine5_translation,
        _combine6_translation,
        _combine7_translation,
    ]


class FPUTWorkbench:
    __slots__ = ("chain_size",)

    def __init__(self, problem_parameters):
        self.chain_size = problem_parameters["chain_size"]

    def __repr__(self) -> str:
        return f"FPUTWorkbench(chain_size={self.chain_size})"

    __str__ = __repr__

    def allocate_state(self):
        return FPUTState(
            np.zeros(self.chain_size, dtype=np.float64),
            np.zeros(self.chain_size, dtype=np.float64),
        )

    def copy_state(self, dst, src):
        np.copyto(dst.q, src.q)
        np.copyto(dst.p, src.p)

    def allocate_translation(self):
        return FPUTTranslation(
            np.zeros(self.chain_size, dtype=np.float64),
            np.zeros(self.chain_size, dtype=np.float64),
        )


class FPUTDerivative:
    __slots__ = ("beta",)

    def __init__(self, problem_parameters):
        self.beta = problem_parameters["beta"]

    def __repr__(self) -> str:
        return f"FPUTDerivative(beta={self.beta!r})"

    __str__ = __repr__

    def __call__(self, state, out):
        if NUMBA_AVAILABLE:
            _rhs_kernel(state.q, state.p, out.dq, out.dp, self.beta)
            return

        q = state.q
        left = np.empty_like(q)
        right = np.empty_like(q)
        left[0] = 0.0
        left[1:] = q[:-1]
        right[-1] = 0.0
        right[:-1] = q[1:]
        out.dq[:] = state.p
        out.dp[:] = right - 2.0 * q + left + self.beta * ((right - q) ** 3 - (q - left) ** 3)


def prepare_rkck(problem_parameters, tolerance_parameters, initial_conditions, reference):
    workbench = FPUTWorkbench(problem_parameters)
    derivative = FPUTDerivative(problem_parameters)
    scheme = SchemeCashKarp(derivative, workbench)
    marcher = Marcher(
        scheme,
        tolerance=Tolerance(atol=tolerance_parameters["atol"], rtol=tolerance_parameters["rtol"]),
    )
    integrate = Integrator()

    def solve_once():
        interval = Interval(
            problem_parameters["t0"],
            tolerance_parameters["initial_step"],
            problem_parameters["t1"],
        )
        state = FPUTState(initial_conditions["q"].copy(), initial_conditions["p"].copy())
        steps = 0

        for _interval, _state in integrate.live(marcher, interval, state):
            steps += 1

        return {
            "library": "STARK",
            "solver": "RKCK",
            "error": state.error_against(reference),
            "steps": steps,
        }

    return solve_once


def run_rkck(problem_parameters, tolerance_parameters, initial_conditions, reference):
    return prepare_rkck(problem_parameters, tolerance_parameters, initial_conditions, reference)()


def prepare_rkdp(problem_parameters, tolerance_parameters, initial_conditions, reference):
    workbench = FPUTWorkbench(problem_parameters)
    derivative = FPUTDerivative(problem_parameters)
    scheme = SchemeDormandPrince(derivative, workbench)
    marcher = Marcher(
        scheme,
        tolerance=Tolerance(atol=tolerance_parameters["atol"], rtol=tolerance_parameters["rtol"]),
    )
    integrate = Integrator()

    def solve_once():
        interval = Interval(
            problem_parameters["t0"],
            tolerance_parameters["initial_step"],
            problem_parameters["t1"],
        )
        state = FPUTState(initial_conditions["q"].copy(), initial_conditions["p"].copy())
        steps = 0

        for _interval, _state in integrate.live(marcher, interval, state):
            steps += 1

        return {
            "library": "STARK",
            "solver": "RKDP",
            "error": state.error_against(reference),
            "steps": steps,
        }

    return solve_once


def run_rkdp(problem_parameters, tolerance_parameters, initial_conditions, reference):
    return prepare_rkdp(problem_parameters, tolerance_parameters, initial_conditions, reference)()
