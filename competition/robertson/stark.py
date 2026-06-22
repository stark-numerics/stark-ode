from __future__ import annotations

from math import acos, copysign, cos, pi, sqrt
from typing import Any

import numpy as np

from stark.core.block import BlockBasis
from stark.diagnostics.comparison import Comparison
from stark.core.configuration import Configuration
from stark.core.interval import Interval
from stark.core.tolerance import Tolerance
from stark.engines import EngineNumpy
from stark.problem import DerivativeStyle
from stark.problem.frame.frame import Frame
from stark.problem.linearizer import LinearizerStyle
from stark.methods.method import Method
from stark.problem.system.system import System
from stark.methods.inverters.dense import InverterDense
from stark.methods.resolvents import (
    ResolventChord,
    ResolventNewton,
    ResolventVeryChord,
)
from stark.methods.schemes import SchemeKvaerno3, SchemeKvaerno4, SchemeKvaerno5


ROBERTSON_LAYOUT = Frame({"y": {"translation": "dy", "shape": (3,)}})

Array = Any


@DerivativeStyle.kernel_accepts_instant_writes(state=("y",), translation=("dy",))
def robertson_rhs(t: float, y: Array, dy: Array) -> None:
    y1 = y[0]
    y2 = y[1]
    y3 = y[2]
    coupling = 1.0e4 * y2 * y3
    quadratic = 3.0e7 * y2 * y2
    dy[0] = -0.04 * y1 + coupling
    dy[1] = 0.04 * y1 - coupling - quadratic
    dy[2] = quadratic


def robertson_jacobian_apply(t: float, state_y: Array, source_dy: Array, out_dy: Array) -> None:
    del t
    y2 = state_y[1]
    y3 = state_y[2]
    coupling = 1.0e4 * (source_dy[1] * y3 + y2 * source_dy[2])
    quadratic = 6.0e7 * y2 * source_dy[1]
    out_dy[0] = -0.04 * source_dy[0] + coupling
    out_dy[1] = 0.04 * source_dy[0] - coupling - quadratic
    out_dy[2] = quadratic


def robertson_jacobian_dense(
    state_y: Array,
    matrix: Array,
    row_offset: int,
    column_offset: int,
    stride: int,
) -> None:
    y2 = state_y[1]
    y3 = state_y[2]

    matrix[(row_offset + 0) * stride + column_offset + 0] = -0.04
    matrix[(row_offset + 0) * stride + column_offset + 1] = 1.0e4 * y3
    matrix[(row_offset + 0) * stride + column_offset + 2] = 1.0e4 * y2

    matrix[(row_offset + 1) * stride + column_offset + 0] = 0.04
    matrix[(row_offset + 1) * stride + column_offset + 1] = -1.0e4 * y3 - 6.0e7 * y2
    matrix[(row_offset + 1) * stride + column_offset + 2] = -1.0e4 * y2

    matrix[(row_offset + 2) * stride + column_offset + 0] = 0.0
    matrix[(row_offset + 2) * stride + column_offset + 1] = 6.0e7 * y2
    matrix[(row_offset + 2) * stride + column_offset + 2] = 0.0



robertson_linearizer = LinearizerStyle.operator(
    apply=robertson_jacobian_apply,
    dense=robertson_jacobian_dense,
    state=("y",),
    source=("dy",),
    target=("dy",),
)

class RobertsonCubicResolvent:
    __slots__ = ("cubic", "tableau")

    def __init__(self, tableau=None) -> None:
        self.tableau = tableau
        self.cubic = RobertsonCubicRoot()

    def bind(self, interval, state) -> None:
        del interval, state

    def __call__(self, problem, out) -> None:
        state_y = problem.origin.y
        alpha = problem.alpha
        rhs = problem.rhs

        delta = out[0]
        if alpha == 0.0:
            if rhs is None:
                delta.dy[:] = 0.0
            else:
                np.copyto(delta.dy, rhs[0].dy)
            return

        rhs_y = rhs[0].dy if rhs is not None else None
        rhs0 = float(rhs_y[0]) if rhs_y is not None else 0.0
        rhs1 = float(rhs_y[1]) if rhs_y is not None else 0.0
        rhs2 = float(rhs_y[2]) if rhs_y is not None else 0.0

        shifted_y2 = float(state_y[1]) + rhs1
        shifted_y3 = float(state_y[2]) + rhs2
        total = float(state_y[0] + state_y[1] + state_y[2] + rhs0 + rhs1 + rhs2)

        coefficient3 = alpha * alpha * 1.0e4 * 3.0e7
        coefficient2 = alpha * 3.0e7 * (1.0 + 0.04 * alpha)
        coefficient1 = 1.0 + 0.04 * alpha + alpha * 1.0e4 * shifted_y3
        coefficient0 = -(shifted_y2 + 0.04 * alpha * (total - shifted_y3))

        z2 = self.cubic.solve(
            coefficient3,
            coefficient2,
            coefficient1,
            coefficient0,
            lower=0.0,
            upper=total,
            target=shifted_y2,
        )
        z3 = shifted_y3 + alpha * 3.0e7 * z2 * z2
        z1 = total - z2 - z3

        delta.dy[0] = z1 - state_y[0]
        delta.dy[1] = z2 - state_y[1]
        delta.dy[2] = z3 - state_y[2]


class RobertsonCubicRoot:
    __slots__ = ()

    @staticmethod
    def cube_root(value: float) -> float:
        if value == 0.0:
            return 0.0
        return copysign(abs(value) ** (1.0 / 3.0), value)

    def solve(self, coefficient3, coefficient2, coefficient1, coefficient0, *, lower, upper, target) -> float:
        a = coefficient2 / coefficient3
        b = coefficient1 / coefficient3
        c = coefficient0 / coefficient3
        p = b - (a * a) / 3.0
        q = (2.0 * a * a * a) / 27.0 - (a * b) / 3.0 + c
        half_q = 0.5 * q
        third_p = p / 3.0
        discriminant = half_q * half_q + third_p * third_p * third_p

        if discriminant >= 0.0:
            root = self.cube_root(-half_q + sqrt(discriminant)) + self.cube_root(-half_q - sqrt(discriminant))
            return root - a / 3.0

        radius = 2.0 * sqrt(-third_p)
        angle = acos(max(-1.0, min(1.0, -half_q / sqrt(-(third_p * third_p * third_p)))))
        roots = [radius * cos((angle + 2.0 * pi * index) / 3.0) - a / 3.0 for index in range(3)]
        bounded = [root for root in roots if lower - 1.0e-12 <= root <= upper + 1.0e-12]
        if bounded:
            return min(bounded, key=lambda root: abs(root - target))
        return min(roots, key=lambda root: abs(root - target))


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
        EngineNumpy(ROBERTSON_LAYOUT)
        if accelerator is None
        else EngineNumpy(ROBERTSON_LAYOUT, accelerator=accelerator)
    )
    linearizer = robertson_linearizer
    system = System(
        derivative=robertson_rhs,
        frame=ROBERTSON_LAYOUT,
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
    scheme=SchemeKvaerno4,
    accelerator=None,
):
    system, engine, linearizer, configuration = stark_runtime(
        stark_parameters,
        accelerator=accelerator,
    )
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


def cubic_resolvent(engine, linearizer, configuration, scheme):
    del engine, linearizer, configuration
    return RobertsonCubicResolvent(tableau=scheme.tableau)


def newton_dense_resolvent(engine, linearizer, configuration, scheme):
    inverter = InverterDense(
        basis=BlockBasis([engine.translation_basis()]),
    )
    return ResolventNewton(
        engine.allocator,
        linearizer=linearizer,
        inverter=inverter,
        configuration=configuration,
        accelerator=engine.accelerator,
        tableau=scheme.tableau,
    )


def chord_dense_resolvent(engine, linearizer, configuration, scheme):
    inverter = InverterDense(
        basis=BlockBasis([engine.translation_basis()]),
    )
    return ResolventChord(
        engine.allocator,
        linearizer=linearizer,
        inverter=inverter,
        configuration=configuration,
        accelerator=engine.accelerator,
        tableau=scheme.tableau,
    )


def very_chord_dense_resolvent(engine, linearizer, configuration, scheme):
    inverter = InverterDense(
        basis=BlockBasis([engine.translation_basis()]),
    )
    return ResolventVeryChord(
        engine.allocator,
        linearizer=linearizer,
        inverter=inverter,
        configuration=configuration,
        accelerator=engine.accelerator,
        tableau=scheme.tableau,
    )


def prepare_kvaerno4_cubic(problem_parameters, stark_parameters, initial_conditions, reference):
    return stark_solver(
        "Kvaerno4 Cubic",
        problem_parameters,
        stark_parameters,
        initial_conditions,
        reference,
        cubic_resolvent,
        scheme=SchemeKvaerno4,
    )


def prepare_kvaerno5_cubic(problem_parameters, stark_parameters, initial_conditions, reference):
    return stark_solver(
        "Kvaerno5 Exact Cubic",
        problem_parameters,
        stark_parameters,
        initial_conditions,
        reference,
        cubic_resolvent,
        scheme=SchemeKvaerno5,
    )


def prepare_kvaerno5_newton_dense(problem_parameters, stark_parameters, initial_conditions, reference):
    return stark_solver(
        "Kvaerno5 Newton Dense",
        problem_parameters,
        stark_parameters,
        initial_conditions,
        reference,
        newton_dense_resolvent,
        scheme=SchemeKvaerno5,
    )


def prepare_kvaerno4_newton_dense_small(problem_parameters, stark_parameters, initial_conditions, reference):
    return stark_solver(
        "Kvaerno4 Newton Dense",
        problem_parameters,
        stark_parameters,
        initial_conditions,
        reference,
        newton_dense_resolvent,
        scheme=SchemeKvaerno4,
    )


def prepare_kvaerno3_newton_dense_small(problem_parameters, stark_parameters, initial_conditions, reference):
    return stark_solver(
        "Kvaerno3 Newton Dense",
        problem_parameters,
        stark_parameters,
        initial_conditions,
        reference,
        newton_dense_resolvent,
        scheme=SchemeKvaerno3,
    )


def prepare_kvaerno5_newton_dense_small(problem_parameters, stark_parameters, initial_conditions, reference):
    return stark_solver(
        "Kvaerno5 Newton Dense",
        problem_parameters,
        stark_parameters,
        initial_conditions,
        reference,
        newton_dense_resolvent,
        scheme=SchemeKvaerno5,
    )


def prepare_kvaerno5_chord_dense_small(problem_parameters, stark_parameters, initial_conditions, reference):
    return stark_solver(
        "Kvaerno5 Chord Dense",
        problem_parameters,
        stark_parameters,
        initial_conditions,
        reference,
        chord_dense_resolvent,
        scheme=SchemeKvaerno5,
    )


def prepare_kvaerno5_very_chord_dense_small(problem_parameters, stark_parameters, initial_conditions, reference):
    return stark_solver(
        "Kvaerno5 VeryChord Dense",
        problem_parameters,
        stark_parameters,
        initial_conditions,
        reference,
        very_chord_dense_resolvent,
        scheme=SchemeKvaerno5,
    )
