from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from stark.diagnostics.comparison import Comparison
from stark.core.configuration import Configuration
from stark.core.interval import Interval
from stark.core.tolerance import Tolerance
from stark.engines import EngineNumpy
from stark.problem import DynamicsStyle
from stark.problem.frame.frame import Frame
from stark.methods.method import Method
from stark.problem.system.system import System
from stark.methods.schemes.explicit.adaptive import SchemeCashKarp, SchemeDormandPrince


Array = Any


@DynamicsStyle.kernel_accepts_instant_writes(state=("q", "p"), translation=("dq", "dp"))
def fput_rhs(
    t: float,
    q: Array,
    p: Array,
    dq: Array,
    dp: Array,
    beta: float,
) -> None:
    size = q.size

    for index in range(size):
        left = 0.0 if index == 0 else q[index - 1]
        right = 0.0 if index == size - 1 else q[index + 1]
        qi = q[index]

        dq[index] = p[index]
        dp[index] = (
            right
            - 2.0 * qi
            + left
            + beta * ((right - qi) ** 3 - (qi - left) ** 3)
        )


class FPUTStarkProblem:
    __slots__ = ("problem_parameters", "system")

    def __init__(self, problem_parameters: Mapping[str, Any]) -> None:
        chain_size = int(problem_parameters["chain_size"])
        self.problem_parameters = problem_parameters
        self.system = System(
            dynamics=fput_rhs.with_parameters(float(problem_parameters["beta"])),
            frame=Frame(
                {
                    "q": {"translation": "dq", "shape": (chain_size,)},
                    "p": {"translation": "dp", "shape": (chain_size,)},
                }
            ),
        )

    def prepare_rkck(
        self,
        problem_parameters: Mapping[str, Any],
        tolerance_parameters: Mapping[str, Any],
        initial_conditions: Mapping[str, Any],
        reference: Mapping[str, Any],
    ):
        del problem_parameters
        return self.stark_solver(
            solver_name="RKCK",
            scheme_type=SchemeCashKarp,
            tolerance_parameters=tolerance_parameters,
            initial_conditions=initial_conditions,
            reference=reference,
        )

    def prepare_rkdp(
        self,
        problem_parameters: Mapping[str, Any],
        tolerance_parameters: Mapping[str, Any],
        initial_conditions: Mapping[str, Any],
        reference: Mapping[str, Any],
    ):
        del problem_parameters
        return self.stark_solver(
            solver_name="RKDP",
            scheme_type=SchemeDormandPrince,
            tolerance_parameters=tolerance_parameters,
            initial_conditions=initial_conditions,
            reference=reference,
        )

    def stark_solver(
        self,
        *,
        solver_name: str,
        scheme_type: type,
        tolerance_parameters: Mapping[str, Any],
        initial_conditions: Mapping[str, Any],
        reference: Mapping[str, Any],
    ):
        ivp = self.system.ivp(
            initial=initial_conditions,
            interval=Interval(
                float(self.problem_parameters["t0"]),
                float(tolerance_parameters["initial_step"]),
                float(self.problem_parameters["t1"]),
            ),
            method=Method(scheme=scheme_type),
            engine=EngineNumpy,
            configuration=Configuration(
                check_progress=False,
                scheme_tolerance=Tolerance(
                    atol=float(tolerance_parameters["atol"]),
                    rtol=float(tolerance_parameters["rtol"]),
                ),
            ),
        )

        def solve_once() -> dict[str, Any]:
            result = ivp.final_result()
            return {
                "library": "STARK",
                "solver": solver_name,
                "error": Comparison.fieldwise_rms_error(
                    result.state,
                    reference,
                    ("q", "p"),
                    sample_count=result.state.q.size,
                ),
                "steps": result.steps,
            }

        return solve_once


def prepare_rkck(
    problem_parameters: Mapping[str, Any],
    tolerance_parameters: Mapping[str, Any],
    initial_conditions: Mapping[str, Any],
    reference: Mapping[str, Any],
):
    return FPUTStarkProblem(problem_parameters).prepare_rkck(
        problem_parameters,
        tolerance_parameters,
        initial_conditions,
        reference,
    )


def prepare_rkdp(
    problem_parameters: Mapping[str, Any],
    tolerance_parameters: Mapping[str, Any],
    initial_conditions: Mapping[str, Any],
    reference: Mapping[str, Any],
):
    return FPUTStarkProblem(problem_parameters).prepare_rkdp(
        problem_parameters,
        tolerance_parameters,
        initial_conditions,
        reference,
    )
