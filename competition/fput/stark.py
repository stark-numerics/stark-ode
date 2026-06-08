from __future__ import annotations

from collections.abc import Callable
from typing import Any

from stark import (
    Configuration,
    Interval,
    StarkDerivative,
    StarkDerivativeStyle,
    StarkLayout,
    StarkLayoutField,
    StarkMethod,
    StarkSystem,
    Tolerance,
)
from stark.accelerators import AcceleratorNone, AcceleratorNumba
from stark.engines import StarkEngineNumpy
from stark.schemes.explicit.adaptive import SchemeCashKarp, SchemeDormandPrince


try:
    ACCELERATOR = AcceleratorNumba()
    USE_NUMBA_ACCELERATION = True
except ModuleNotFoundError:
    ACCELERATOR = AcceleratorNone()
    USE_NUMBA_ACCELERATION = False


Array = Any


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


class FPUTStarkRunner:
    __slots__ = (
        "ivp",
        "problem_parameters",
        "reference",
        "solver_name",
        "tolerance_parameters",
    )

    def __init__(
        self,
        *,
        system: StarkSystem,
        engine: Callable[[StarkLayout], StarkEngineNumpy],
        solver_name: str,
        method: StarkMethod,
        tolerance_parameters: dict[str, Any],
        initial_conditions: dict[str, Array],
        problem_parameters: dict[str, Any],
        reference: dict[str, Array],
    ) -> None:
        self.ivp = system.ivp(
            initial=initial_conditions,
            interval=self.make_interval(problem_parameters, tolerance_parameters),
            method=method,
            engine=engine,
            configuration=Configuration(
                check_progress=False,
                scheme_tolerance=Tolerance(
                    atol=float(tolerance_parameters["atol"]),
                    rtol=float(tolerance_parameters["rtol"]),
                ),
            ),
        )
        self.problem_parameters = problem_parameters
        self.reference = reference
        self.solver_name = solver_name
        self.tolerance_parameters = tolerance_parameters

    @staticmethod
    def make_interval(
        problem_parameters: dict[str, Any],
        tolerance_parameters: dict[str, Any],
    ) -> Interval:
        return Interval(
            float(problem_parameters["t0"]),
            float(tolerance_parameters["initial_step"]),
            float(problem_parameters["t1"]),
        )

    def __call__(self) -> dict[str, Any]:
        interval = self.make_interval(self.problem_parameters, self.tolerance_parameters)
        state = self.ivp.engine.allocator.allocate_state()
        self.ivp.engine.allocator.copy_state(self.ivp.initial, state)

        steps = 0
        for _interval, _state in self.ivp.integrator.mutating_trajectory(self.ivp.stepper, interval, state):
            steps += 1

        return {
            "library": "STARK",
            "solver": self.solver_name,
            "error": float(
                _state_error_kernel(
                    state.q,
                    state.p,
                    self.reference["q"],
                    self.reference["p"],
                )
            ),
            "steps": steps,
        }


class FPUTStarkProblem:
    __slots__ = ("problem_parameters", "system")

    def __init__(self, problem_parameters: dict[str, Any]) -> None:
        chain_size = int(problem_parameters["chain_size"])
        layout = StarkLayout(
            (
                StarkLayoutField("q", translation="dq", shape=(chain_size,)),
                StarkLayoutField("p", translation="dp", shape=(chain_size,)),
            )
        )

        self.problem_parameters = problem_parameters
        derivative = StarkDerivative(
            StarkDerivativeStyle.kernel(
                _rhs_kernel,
                state=("q", "p"),
                translation=("dq", "dp"),
                parameters=(float(problem_parameters["beta"]),),
            )
        )
        self.system = StarkSystem(
            derivative=derivative,
            layout=layout,
        )

    def make_engine(self, layout: StarkLayout) -> StarkEngineNumpy:
        return StarkEngineNumpy(layout, accelerator=ACCELERATOR)

    def prepare_rkck(
        self,
        problem_parameters: dict[str, Any],
        tolerance_parameters: dict[str, Any],
        initial_conditions: dict[str, Array],
        reference: dict[str, Array],
    ) -> FPUTStarkRunner:
        del problem_parameters
        return self.prepare(
            solver_name="RKCK",
            scheme_type=SchemeCashKarp,
            tolerance_parameters=tolerance_parameters,
            initial_conditions=initial_conditions,
            reference=reference,
        )

    def prepare_rkdp(
        self,
        problem_parameters: dict[str, Any],
        tolerance_parameters: dict[str, Any],
        initial_conditions: dict[str, Array],
        reference: dict[str, Array],
    ) -> FPUTStarkRunner:
        del problem_parameters
        return self.prepare(
            solver_name="RKDP",
            scheme_type=SchemeDormandPrince,
            tolerance_parameters=tolerance_parameters,
            initial_conditions=initial_conditions,
            reference=reference,
        )

    def prepare(
        self,
        *,
        solver_name: str,
        scheme_type: type,
        tolerance_parameters: dict[str, Any],
        initial_conditions: dict[str, Array],
        reference: dict[str, Array],
    ) -> FPUTStarkRunner:
        return FPUTStarkRunner(
            system=self.system,
            engine=self.make_engine,
            solver_name=solver_name,
            method=StarkMethod(scheme=scheme_type),
            problem_parameters=self.problem_parameters,
            tolerance_parameters=tolerance_parameters,
            initial_conditions=initial_conditions,
            reference=reference,
        )


def prepare_rkck(
    problem_parameters: dict[str, Any],
    tolerance_parameters: dict[str, Any],
    initial_conditions: dict[str, Array],
    reference: dict[str, Array],
) -> FPUTStarkRunner:
    return FPUTStarkProblem(problem_parameters).prepare_rkck(
        problem_parameters,
        tolerance_parameters,
        initial_conditions,
        reference,
    )


def prepare_rkdp(
    problem_parameters: dict[str, Any],
    tolerance_parameters: dict[str, Any],
    initial_conditions: dict[str, Array],
    reference: dict[str, Array],
) -> FPUTStarkRunner:
    return FPUTStarkProblem(problem_parameters).prepare_rkdp(
        problem_parameters,
        tolerance_parameters,
        initial_conditions,
        reference,
    )


__all__ = [
    "FPUTStarkProblem",
    "FPUTStarkRunner",
    "USE_NUMBA_ACCELERATION",
    "prepare_rkck",
    "prepare_rkdp",
]
