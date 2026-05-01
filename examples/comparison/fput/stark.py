from __future__ import annotations

from math import sqrt

import numpy as np

from stark import Executor, Marcher, Integrator, Interval, Safety, Tolerance
from stark.accelerators import Accelerator
from stark.algebraist import Algebraist, AlgebraistField, AlgebraistLooped
from stark.schemes.explicit_adaptive import SchemeCashKarp, SchemeDormandPrince


try:
    ACCELERATOR = Accelerator.numba()
    USE_NUMBA_ACCELERATION = True
except ModuleNotFoundError:
    ACCELERATOR = Accelerator.none()
    USE_NUMBA_ACCELERATION = False
ALGEBRAIST = Algebraist(
    fields=(
        AlgebraistField("dq", "q", policy=AlgebraistLooped(rank=1)),
        AlgebraistField("dp", "p", policy=AlgebraistLooped(rank=1)),
    ),
    accelerator=ACCELERATOR,
    generate_norm="l2",
)

@ACCELERATOR.decorate
def _rhs_kernel(q, p, dq, dp, beta):
    size = q.size
    for i in range(size):
        left = 0.0 if i == 0 else q[i - 1]
        right = 0.0 if i == size - 1 else q[i + 1]
        qi = q[i]
        dq[i] = p[i]
        dp[i] = right - 2.0 * qi + left + beta * ((right - qi) ** 3 - (qi - left) ** 3)

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

    def norm(self):
        return float(ALGEBRAIST.norm(self) / sqrt(self.dq.size))

    def __add__(self, other):
        return FPUTTranslation(self.dq + other.dq, self.dp + other.dp)

    def __rmul__(self, scalar):
        return FPUTTranslation(scalar * self.dq, scalar * self.dp)

    linear_combine = ALGEBRAIST.linear_combine
    __call__ = ALGEBRAIST.apply


class FPUTWorkbench:
    __slots__ = ("chain_size",)
    _compiled = False

    def __init__(self, problem_parameters):
        self.chain_size = problem_parameters["chain_size"]
        if not self.__class__._compiled:
            probe = np.zeros(self.chain_size, dtype=np.float64)
            ALGEBRAIST.compile_examples(probe, probe)
            self.__class__._compiled = True

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
    _compiled = False

    def __init__(self, problem_parameters):
        self.beta = problem_parameters["beta"]
        if not self.__class__._compiled:
            probe = np.zeros(problem_parameters["chain_size"], dtype=np.float64)
            ACCELERATOR.compile_examples(_rhs_kernel, (probe, probe, probe, probe, self.beta))
            self.__class__._compiled = True

    def __repr__(self) -> str:
        return f"FPUTDerivative(beta={self.beta!r})"

    __str__ = __repr__

    def __call__(self, interval, state, out):
        del interval
        if USE_NUMBA_ACCELERATION:
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
    safety = Safety.fast()
    workbench = FPUTWorkbench(problem_parameters)
    derivative = FPUTDerivative(problem_parameters)
    scheme = SchemeCashKarp(derivative, workbench)
    marcher = Marcher(
        scheme,
        Executor(
            tolerance=Tolerance(atol=tolerance_parameters["atol"], rtol=tolerance_parameters["rtol"]),
            safety=safety,
            accelerator=ACCELERATOR,
        ),
    )
    integrate = Integrator(executor=Executor(safety=safety, accelerator=ACCELERATOR))

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
    safety = Safety.fast()
    workbench = FPUTWorkbench(problem_parameters)
    derivative = FPUTDerivative(problem_parameters)
    scheme = SchemeDormandPrince(derivative, workbench)
    marcher = Marcher(
        scheme,
        Executor(
            tolerance=Tolerance(atol=tolerance_parameters["atol"], rtol=tolerance_parameters["rtol"]),
            safety=safety,
            accelerator=ACCELERATOR,
        ),
    )
    integrate = Integrator(executor=Executor(safety=safety, accelerator=ACCELERATOR))

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
















