from __future__ import annotations

from typing import Any

import numpy as np

from stark.core.configuration import Configuration
from stark.core.interval import Interval
from stark.core.tolerance import Tolerance
from stark.engines.numpy.engine import StarkEngineNumpy
from stark.interface.derivative import StarkDerivativeStyle
from stark.interface.layout import StarkLayout
from stark.interface.method import StarkMethod
from stark.interface.system import StarkSystem
from stark.schemes.explicit.adaptive.cash_karp import SchemeCashKarp
from stark.schemes.explicit.adaptive.dormand_prince import SchemeDormandPrince


Array = Any


@StarkDerivativeStyle.kernel(state=("u", "v"), translation=("du", "dv"))
def brusselator_rhs(
    u: Array,
    v: Array,
    du: Array,
    dv: Array,
    alpha: float,
    a: float,
    b: float,
    inv_dx2: float,
) -> None:
    rows, cols = u.shape
    for i in range(rows):
        im1 = rows - 1 if i == 0 else i - 1
        ip1 = 0 if i == rows - 1 else i + 1
        for j in range(cols):
            jm1 = cols - 1 if j == 0 else j - 1
            jp1 = 0 if j == cols - 1 else j + 1
            u_ij = u[i, j]
            v_ij = v[i, j]
            reaction = u_ij * u_ij * v_ij
            lap_u = (
                u[im1, j] + u[ip1, j] + u[i, jm1] + u[i, jp1] - 4.0 * u_ij
            ) * inv_dx2
            lap_v = (
                v[im1, j] + v[ip1, j] + v[i, jm1] + v[i, jp1] - 4.0 * v_ij
            ) * inv_dx2
            du[i, j] = alpha * lap_u + a + reaction - (b + 1.0) * u_ij
            dv[i, j] = alpha * lap_v + b * u_ij - reaction


def prepare_solver(
    solver_name,
    scheme_type,
    problem_parameters,
    tolerance_parameters,
    initial_conditions,
    reference,
):
    grid_shape = (int(problem_parameters["grid_size"]), int(problem_parameters["grid_size"]))
    t0 = float(problem_parameters["t0"])
    initial_step = float(tolerance_parameters["initial_step"])
    t1 = float(problem_parameters["t1"])
    configuration = Configuration(
        check_progress=False,
        scheme_tolerance=Tolerance(
            atol=float(tolerance_parameters["atol"]),
            rtol=float(tolerance_parameters["rtol"]),
        ),
    )
    system = StarkSystem(
        derivative=brusselator_rhs.with_parameters(
            float(problem_parameters["alpha"]),
            float(problem_parameters["a"]),
            float(problem_parameters["b"]),
            float(problem_parameters["inv_dx2"]),
        ),
        layout=StarkLayout(
            {
                "u": {"translation": "du", "shape": grid_shape},
                "v": {"translation": "dv", "shape": grid_shape},
            }
        ),
    )
    ivp = system.ivp(
        initial=initial_conditions,
        interval=Interval(t0, initial_step, t1),
        method=StarkMethod(scheme=scheme_type),
        engine=StarkEngineNumpy,
        configuration=configuration,
    )

    def solve_once() -> dict[str, Any]:
        interval = ivp.fresh_interval()
        state = ivp.fresh_state()
        steps = 0

        for _interval, _state in ivp.mutating_trajectory(interval=interval, state=state):
            steps += 1

        du = state.u - reference["u"]
        dv = state.v - reference["v"]
        error = np.sqrt(
            (np.dot(du.ravel(), du.ravel()) + np.dot(dv.ravel(), dv.ravel()))
            / state.u.size
        )

        return {
            "library": "STARK",
            "solver": solver_name,
            "error": float(error),
            "steps": steps,
        }

    return solve_once


def prepare_rkck(problem_parameters, tolerance_parameters, initial_conditions, reference):
    return prepare_solver(
        "RKCK",
        SchemeCashKarp,
        problem_parameters,
        tolerance_parameters,
        initial_conditions,
        reference,
    )


def run_rkck(problem_parameters, tolerance_parameters, initial_conditions, reference):
    return prepare_rkck(problem_parameters, tolerance_parameters, initial_conditions, reference)()


def prepare_rkdp(problem_parameters, tolerance_parameters, initial_conditions, reference):
    return prepare_solver(
        "RKDP",
        SchemeDormandPrince,
        problem_parameters,
        tolerance_parameters,
        initial_conditions,
        reference,
    )


def run_rkdp(problem_parameters, tolerance_parameters, initial_conditions, reference):
    return prepare_rkdp(problem_parameters, tolerance_parameters, initial_conditions, reference)()
