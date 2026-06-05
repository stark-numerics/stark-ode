from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from stark import Configuration, Integrator, Interval, IntegratorStepper, Tolerance
from stark.accelerators import AcceleratorNone, AcceleratorNumba
from stark.algebraist.arity import AlgebraistArity
from stark.algebraist.generator import AlgebraistGeneratorGeneral
from stark.algebraist.layout import (
    AlgebraistLayout,
    AlgebraistLayoutField,
    AlgebraistLayoutLooped,
)
from stark.schemes.explicit.adaptive import SchemeCashKarp, SchemeDormandPrince


try:
    ACCELERATOR = AcceleratorNumba()
    USE_NUMBA_ACCELERATION = True
except ModuleNotFoundError:
    ACCELERATOR = AcceleratorNone()
    USE_NUMBA_ACCELERATION = False


Array = Any


FPUT_LAYOUT = AlgebraistLayout(
    fields=(
        AlgebraistLayoutField(
            translation_path="dq",
            state_path="q",
            policy=AlgebraistLayoutLooped(rank=1),
        ),
        AlgebraistLayoutField(
            translation_path="dp",
            state_path="p",
            policy=AlgebraistLayoutLooped(rank=1),
        ),
    )
)


@ACCELERATOR.compile
def _rhs_kernel(
    q: Array,
    p: Array,
    dq: Array,
    dp: Array,
    beta: float,
) -> None:
    size = q.size

    for i in range(size):
        left = 0.0 if i == 0 else q[i - 1]
        right = 0.0 if i == size - 1 else q[i + 1]
        qi = q[i]

        dq[i] = p[i]
        dp[i] = (
            right
            - 2.0 * qi
            + left
            + beta * ((right - qi) ** 3 - (qi - left) ** 3)
        )


@ACCELERATOR.compile
def _apply_kernel(
    dq: Array,
    dp: Array,
    origin_q: Array,
    origin_p: Array,
    result_q: Array,
    result_p: Array,
) -> None:
    size = dq.size

    for i in range(size):
        result_q[i] = origin_q[i] + dq[i]
        result_p[i] = origin_p[i] + dp[i]


@ACCELERATOR.compile
def _norm_kernel(dq: Array, dp: Array) -> float:
    size = dq.size
    total = 0.0

    for i in range(size):
        total += dq[i] * dq[i] + dp[i] * dp[i]

    return (total / size) ** 0.5


@ACCELERATOR.compile
def _state_error_kernel(
    q: Array,
    p: Array,
    reference_q: Array,
    reference_p: Array,
) -> float:
    size = q.size
    total = 0.0

    for i in range(size):
        q_error = q[i] - reference_q[i]
        p_error = p[i] - reference_p[i]
        total += q_error * q_error + p_error * p_error

    return (total / size) ** 0.5


class FPUTState:
    __slots__ = ("q", "p")

    def __init__(self, q: Array, p: Array) -> None:
        self.q = q
        self.p = p

    def __repr__(self) -> str:
        return f"FPUTState(size={self.q.size})"

    __str__ = __repr__

    def error_against(self, reference: dict[str, Array]) -> float:
        return float(
            _state_error_kernel(
                self.q,
                self.p,
                reference["q"],
                reference["p"],
            )
        )


class FPUTTranslation:
    __slots__ = ("dq", "dp")

    linear_combine: tuple[Callable[..., Any], ...] = ()

    def __init__(self, dq: Array, dp: Array) -> None:
        self.dq = dq
        self.dp = dp

    def __repr__(self) -> str:
        return f"FPUTTranslation(size={self.dq.size})"

    __str__ = __repr__

    def norm(self) -> float:
        return float(_norm_kernel(self.dq, self.dp))

    def __add__(self, other: FPUTTranslation) -> FPUTTranslation:
        return FPUTTranslation(
            self.dq + other.dq,
            self.dp + other.dp,
        )

    def __sub__(self, other: FPUTTranslation) -> FPUTTranslation:
        return FPUTTranslation(
            self.dq - other.dq,
            self.dp - other.dp,
        )

    def __rmul__(self, scalar: float) -> FPUTTranslation:
        return FPUTTranslation(
            scalar * self.dq,
            scalar * self.dp,
        )

    def __mul__(self, scalar: float) -> FPUTTranslation:
        return self.__rmul__(scalar)

    def __call__(self, origin: FPUTState, result: FPUTState) -> None:
        _apply_kernel(
            self.dq,
            self.dp,
            origin.q,
            origin.p,
            result.q,
            result.p,
        )


class FPUTAllocator:
    __slots__ = ("chain_size",)

    _algebraist_installed = False

    def __init__(self, problem_parameters: dict[str, Any]) -> None:
        self.chain_size = int(problem_parameters["chain_size"])
        self._install_algebraist_linear_combine()

    def __repr__(self) -> str:
        return f"FPUTAllocator(chain_size={self.chain_size})"

    __str__ = __repr__

    def allocate_state(self) -> FPUTState:
        return FPUTState(
            np.zeros(self.chain_size, dtype=np.float64),
            np.zeros(self.chain_size, dtype=np.float64),
        )

    def copy_state(self, source: FPUTState, out: FPUTState) -> None:
        np.copyto(out.q, source.q)
        np.copyto(out.p, source.p)

    def allocate_translation(self) -> FPUTTranslation:
        return FPUTTranslation(
            np.zeros(self.chain_size, dtype=np.float64),
            np.zeros(self.chain_size, dtype=np.float64),
        )

    def _install_algebraist_linear_combine(self) -> None:
        if self.__class__._algebraist_installed:
            return

        provider = AlgebraistGeneratorGeneral(
            translation=self.allocate_translation(),
            allocator=self,
            layout=FPUT_LAYOUT,
            accelerator=ACCELERATOR,
        )

        FPUTTranslation.linear_combine = tuple(
            provider.provide(AlgebraistArity(arity))
            for arity in range(1, 13)
        )
        self.__class__._algebraist_installed = True


class FPUTDerivative:
    __slots__ = ("beta",)

    def __init__(self, problem_parameters: dict[str, Any]) -> None:
        self.beta = float(problem_parameters["beta"])

    def __repr__(self) -> str:
        return f"FPUTDerivative(beta={self.beta!r})"

    __str__ = __repr__

    def __call__(
        self,
        interval: Interval,
        state: FPUTState,
        out: FPUTTranslation,
    ) -> None:
        _rhs_kernel(
            state.q,
            state.p,
            out.dq,
            out.dp,
            self.beta,
        )


def _make_state(initial_conditions: dict[str, Array]) -> FPUTState:
    return FPUTState(
        np.asarray(initial_conditions["q"], dtype=np.float64).copy(),
        np.asarray(initial_conditions["p"], dtype=np.float64).copy(),
    )


def _prepare_stark_runner(
    *,
    solver_name: str,
    scheme_type: type,
    problem_parameters: dict[str, Any],
    tolerance_parameters: dict[str, Any],
    initial_conditions: dict[str, Array],
    reference: dict[str, Array],
) -> Callable[[], dict[str, Any]]:
    configuration = Configuration(check_progress=False)

    allocator = FPUTAllocator(problem_parameters)
    derivative = FPUTDerivative(problem_parameters)

    configuration = Configuration(
        scheme_tolerance=Tolerance(
            atol=float(tolerance_parameters["atol"]),
            rtol=float(tolerance_parameters["rtol"]),
        ),
    )
    scheme = scheme_type(derivative, allocator, configuration=configuration)

    stepper = IntegratorStepper(scheme)
    integrate = Integrator(configuration=Configuration(check_progress=False))

    def solve_once() -> dict[str, Any]:
        interval = Interval(
            float(problem_parameters["t0"]),
            float(tolerance_parameters["initial_step"]),
            float(problem_parameters["t1"]),
        )
        state = _make_state(initial_conditions)

        steps = 0
        for _interval, _state in integrate.live(stepper, interval, state):
            steps += 1

        return {
            "library": "STARK",
            "solver": solver_name,
            "error": state.error_against(reference),
            "steps": steps,
        }

    return solve_once


def prepare_rkck(
    problem_parameters: dict[str, Any],
    tolerance_parameters: dict[str, Any],
    initial_conditions: dict[str, Array],
    reference: dict[str, Array],
) -> Callable[[], dict[str, Any]]:
    return _prepare_stark_runner(
        solver_name="RKCK",
        scheme_type=SchemeCashKarp,
        problem_parameters=problem_parameters,
        tolerance_parameters=tolerance_parameters,
        initial_conditions=initial_conditions,
        reference=reference,
    )


def run_rkck(
    problem_parameters: dict[str, Any],
    tolerance_parameters: dict[str, Any],
    initial_conditions: dict[str, Array],
    reference: dict[str, Array],
) -> dict[str, Any]:
    return prepare_rkck(
        problem_parameters,
        tolerance_parameters,
        initial_conditions,
        reference,
    )()


def prepare_rkdp(
    problem_parameters: dict[str, Any],
    tolerance_parameters: dict[str, Any],
    initial_conditions: dict[str, Array],
    reference: dict[str, Array],
) -> Callable[[], dict[str, Any]]:
    return _prepare_stark_runner(
        solver_name="RKDP",
        scheme_type=SchemeDormandPrince,
        problem_parameters=problem_parameters,
        tolerance_parameters=tolerance_parameters,
        initial_conditions=initial_conditions,
        reference=reference,
    )


def run_rkdp(
    problem_parameters: dict[str, Any],
    tolerance_parameters: dict[str, Any],
    initial_conditions: dict[str, Array],
    reference: dict[str, Array],
) -> dict[str, Any]:
    return prepare_rkdp(
        problem_parameters,
        tolerance_parameters,
        initial_conditions,
        reference,
    )()


__all__ = [
    "FPUTDerivative",
    "FPUT_LAYOUT",
    "FPUTState",
    "FPUTTranslation",
    "FPUTAllocator",
    "USE_NUMBA_ACCELERATION",
    "prepare_rkck",
    "prepare_rkdp",
    "run_rkck",
    "run_rkdp",
]
