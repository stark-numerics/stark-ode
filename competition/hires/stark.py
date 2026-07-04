from __future__ import annotations

from typing import Any

import numpy as np

from stark.core.block import BlockBasis
from stark.core.configuration import Configuration
from stark.core.interval import Interval
from stark.core.tolerance import Tolerance
from stark.diagnostics.comparison import Comparison
from stark.engines import EngineNumpy
from stark.methods.inverters.dense import InverterDense
from stark.methods.method import Method
from stark.methods.resolvents import ResolventChord, ResolventNewton, ResolventVeryChord
from stark.methods.schemes import SchemeKvaerno5
from stark.problem import DynamicsStyle
from stark.problem.frame.frame import Frame
from stark.problem.linearizer import LinearizerStyle
from stark.problem.system.system import System


HIRES_FRAME = Frame({"y": {"translation": "dy", "shape": (8,)}})
Array = Any


@DynamicsStyle.kernel_accepts_instant_writes(state=("y",), translation=("dy",))
def hires_rhs(t: float, y: Array, dy: Array) -> None:
    reaction = 280.0 * y[5] * y[7]
    dy[0] = -1.71 * y[0] + 0.43 * y[1] + 8.32 * y[2] + 0.0007
    dy[1] = 1.71 * y[0] - 8.75 * y[1]
    dy[2] = -10.03 * y[2] + 0.43 * y[3] + 0.035 * y[4]
    dy[3] = 8.32 * y[1] + 1.71 * y[2] - 1.12 * y[3]
    dy[4] = -1.745 * y[4] + 0.43 * y[5] + 0.43 * y[6]
    dy[5] = -reaction + 0.69 * y[3] + 1.71 * y[4] - 0.43 * y[5] + 0.69 * y[6]
    dy[6] = reaction - 1.81 * y[6]
    dy[7] = -reaction + 1.81 * y[6]


def hires_jacobian_apply(t: float, state_y: Array, source_dy: Array, out_dy: Array) -> None:
    del t
    y5 = state_y[5]
    y7 = state_y[7]
    source5 = source_dy[5]
    source7 = source_dy[7]
    reaction = 280.0 * (source5 * y7 + y5 * source7)

    out_dy[0] = -1.71 * source_dy[0] + 0.43 * source_dy[1] + 8.32 * source_dy[2]
    out_dy[1] = 1.71 * source_dy[0] - 8.75 * source_dy[1]
    out_dy[2] = -10.03 * source_dy[2] + 0.43 * source_dy[3] + 0.035 * source_dy[4]
    out_dy[3] = 8.32 * source_dy[1] + 1.71 * source_dy[2] - 1.12 * source_dy[3]
    out_dy[4] = -1.745 * source_dy[4] + 0.43 * source_dy[5] + 0.43 * source_dy[6]
    out_dy[5] = -reaction + 0.69 * source_dy[3] + 1.71 * source_dy[4] - 0.43 * source_dy[5] + 0.69 * source_dy[6]
    out_dy[6] = reaction - 1.81 * source_dy[6]
    out_dy[7] = -reaction + 1.81 * source_dy[6]


def hires_jacobian_dense(
    state_y: Array,
    matrix: Array,
    row_offset: int,
    column_offset: int,
    stride: int,
) -> None:
    y5 = state_y[5]
    y7 = state_y[7]
    entries = (
        (0, 0, -1.71), (0, 1, 0.43), (0, 2, 8.32),
        (1, 0, 1.71), (1, 1, -8.75),
        (2, 2, -10.03), (2, 3, 0.43), (2, 4, 0.035),
        (3, 1, 8.32), (3, 2, 1.71), (3, 3, -1.12),
        (4, 4, -1.745), (4, 5, 0.43), (4, 6, 0.43),
        (5, 3, 0.69), (5, 4, 1.71), (5, 5, -280.0 * y7 - 0.43),
        (5, 6, 0.69), (5, 7, -280.0 * y5),
        (6, 5, 280.0 * y7), (6, 6, -1.81), (6, 7, 280.0 * y5),
        (7, 5, -280.0 * y7), (7, 6, 1.81), (7, 7, -280.0 * y5),
    )
    for row, column, value in entries:
        matrix[(row_offset + row) * stride + column_offset + column] = value



hires_linearizer = LinearizerStyle.operator(
    apply=hires_jacobian_apply,
    dense=hires_jacobian_dense,
    state=("y",),
    source=("dy",),
    target=("dy",),
)

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


def stark_runtime(stark_parameters, accelerator=None):
    engine = (
        EngineNumpy(HIRES_FRAME)
        if accelerator is None
        else EngineNumpy(HIRES_FRAME, accelerator=accelerator)
    )
    linearizer = hires_linearizer
    system = System(
        dynamics=hires_rhs,
        frame=HIRES_FRAME,
        linearizer=linearizer,
    )
    prepared_linearizer = system.prepare_linearizer(engine)
    return system, engine, prepared_linearizer, stark_configuration(stark_parameters)


def stark_solver(
    name: str,
    problem_parameters,
    stark_parameters,
    initial_conditions,
    reference,
    resolvent_builder,
    scheme=SchemeKvaerno5,
    accelerator=None,
):
    system, engine, linearizer, configuration = stark_runtime(stark_parameters, accelerator=accelerator)
    resolvent = resolvent_builder(engine, linearizer, configuration, scheme)
    ivp = system.ivp(
        initial=initial_conditions,
        interval=Interval(problem_parameters["t0"], stark_parameters["step"], problem_parameters["t1"]),
        method=Method(scheme=scheme, resolvent=resolvent),
        engine=lambda frame: engine,
        configuration=configuration,
    )

    def solve_once() -> dict[str, Any]:
        result = ivp.final_result()
        return {
            "library": "STARK",
            "solver": name,
            "error": Comparison.fieldwise_rms_error(result.state, reference, ("y",)),
            "steps": result.steps,
        }

    return solve_once


def newton_dense_resolvent(engine, linearizer, configuration, scheme):
    inverter = InverterDense(basis=BlockBasis([engine.translation_basis()]))
    return ResolventNewton(
        engine.allocator,
        linearizer=linearizer,
        inverter=inverter,
        configuration=configuration,
        accelerator=engine.accelerator,
        tableau=scheme.tableau,
    )


def chord_dense_resolvent(engine, linearizer, configuration, scheme):
    inverter = InverterDense(basis=BlockBasis([engine.translation_basis()]))
    return ResolventChord(
        engine.allocator,
        linearizer=linearizer,
        inverter=inverter,
        configuration=configuration,
        accelerator=engine.accelerator,
        tableau=scheme.tableau,
    )


def very_chord_dense_resolvent(engine, linearizer, configuration, scheme):
    inverter = InverterDense(basis=BlockBasis([engine.translation_basis()]))
    return ResolventVeryChord(
        engine.allocator,
        linearizer=linearizer,
        inverter=inverter,
        configuration=configuration,
        accelerator=engine.accelerator,
        tableau=scheme.tableau,
    )


def prepare_kvaerno5_newton_dense(problem_parameters, stark_parameters, initial_conditions, reference):
    return stark_solver(
        "Kvaerno5 Newton Dense",
        problem_parameters,
        stark_parameters,
        initial_conditions,
        reference,
        newton_dense_resolvent,
    )


def prepare_kvaerno5_chord_dense(problem_parameters, stark_parameters, initial_conditions, reference):
    return stark_solver(
        "Kvaerno5 Chord Dense",
        problem_parameters,
        stark_parameters,
        initial_conditions,
        reference,
        chord_dense_resolvent,
    )


def prepare_kvaerno5_very_chord_dense(problem_parameters, stark_parameters, initial_conditions, reference):
    return stark_solver(
        "Kvaerno5 VeryChord Dense",
        problem_parameters,
        stark_parameters,
        initial_conditions,
        reference,
        very_chord_dense_resolvent,
    )
