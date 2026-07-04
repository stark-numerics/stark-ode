from __future__ import annotations

from typing import Any

import numpy as np

from stark.core.configuration import Configuration
from stark.core.interval import Interval
from stark.core.tolerance import Tolerance
from stark.diagnostics.comparison import Comparison
from stark.engines import EngineNumpy
from stark.methods.inverters.krylov import InverterKrylovArnoldi
from stark.methods.method import Method
from stark.methods.resolvents import ResolventNewton
from stark.methods.schemes import SchemeSDIRK21
from stark.problem import DynamicsStyle
from stark.problem.frame.frame import Frame
from stark.problem.linearizer import LinearizerStyle
from competition.allen_cahn_1d.operator import (
    AllenCahnJacobianPeriodicTridiagonal,
    AllenCahnPreconditionerPeriodicTridiagonal,
)
from stark.problem.system.system import System


Array = Any


def _dx(problem_parameters) -> float:
    return problem_parameters["length"] / problem_parameters["grid_size"]


def frame(problem_parameters) -> Frame:
    return Frame({"u": {"translation": "du", "shape": (problem_parameters["grid_size"],)}})


def make_dynamics(problem_parameters):
    diffusivity = problem_parameters["diffusivity"]
    inv_dx2 = 1.0 / (_dx(problem_parameters) ** 2)

    @DynamicsStyle.kernel_accepts_instant_writes(state=("u",), translation=("du",))
    def dynamics(t: float, u: Array, du: Array) -> None:
        du[:] = diffusivity * (np.roll(u, 1) - 2.0 * u + np.roll(u, -1)) * inv_dx2 + u - u * u * u

    return dynamics


def make_linearizer(problem_parameters):
    diffusivity = problem_parameters["diffusivity"]
    inv_dx2 = 1.0 / (_dx(problem_parameters) ** 2)

    def linearizer(_interval, state, out) -> None:
        out.apply = AllenCahnJacobianPeriodicTridiagonal(
            state.u,
            diffusivity=diffusivity,
            inv_dx2=inv_dx2,
        )

    return LinearizerStyle.accepts_interval_writes(linearizer)


def inner_product(left, right) -> float:
    return float(np.dot(left.du, right.du))


def stark_configuration(stark_parameters) -> Configuration:
    return Configuration(
        check_progress=False,
        scheme_tolerance=Tolerance(
            atol=stark_parameters["tolerance_atol"],
            rtol=stark_parameters["tolerance_rtol"],
        ),
        resolvent_tolerance=Tolerance(
            atol=stark_parameters["resolution_atol"],
            rtol=stark_parameters["resolution_rtol"],
        ),
        resolvent_maximum_steps=stark_parameters["resolution_max_iterations"],
        inverter_tolerance=Tolerance(
            atol=stark_parameters["inversion_atol"],
            rtol=stark_parameters["inversion_rtol"],
        ),
        inverter_maximum_steps=stark_parameters["inversion_max_iterations"],
    )


def prepare_sdirk21_newton_krylov(problem_parameters, stark_parameters, initial_conditions, reference):
    problem_frame = frame(problem_parameters)
    engine = EngineNumpy(problem_frame)
    system = System(
        dynamics=make_dynamics(problem_parameters),
        frame=problem_frame,
        linearizer=make_linearizer(problem_parameters),
    )
    linearizer = system.prepare_linearizer(engine)
    if linearizer is None:
        raise RuntimeError("Allen-Cahn competition requires a configured linearizer.")
    configuration = stark_configuration(stark_parameters)
    inverter = InverterKrylovArnoldi(
        engine.allocator,
        inner_product,
        restart=stark_parameters["inversion_restart"],
        configuration=configuration,
        accelerator=engine.accelerator,
        preconditioner=AllenCahnPreconditionerPeriodicTridiagonal(),
    )
    resolvent = ResolventNewton(
        engine.allocator,
        linearizer=linearizer,
        inverter=inverter,
        configuration=configuration,
        accelerator=engine.accelerator,
        tableau=SchemeSDIRK21.tableau,
    )
    ivp = system.ivp(
        initial=initial_conditions,
        interval=Interval(problem_parameters["t0"], stark_parameters["step"], problem_parameters["t1"]),
        method=Method(scheme=SchemeSDIRK21, resolvent=resolvent),
        engine=lambda _frame: engine,
        configuration=configuration,
    )

    def solve_once() -> dict[str, Any]:
        result = ivp.final_result()
        return {
            "library": "STARK",
            "solver": "SDIRK21 Newton Krylov",
            "error": Comparison.fieldwise_rms_error(result.state, reference, ("u",)),
            "steps": result.steps,
        }

    return solve_once
