from __future__ import annotations

from math import acos, copysign, cos, pi, sqrt
from typing import Any

import numpy as np

from stark.block import BlockBasis, BlockSpecialist
from stark.comparison import Comparison
from stark.core.configuration import Configuration
from stark.core.interval import Interval
from stark.core.tolerance import Tolerance
from stark.engines import EngineNumpy
from stark.problem.derivative.derivative import DerivativeStyle
from stark.problem.frame.frame import Frame
from stark.methods.method import Method
from stark.problem.system.system import System
from stark.methods.inverters import InverterRelaxationJacobi
from stark.methods.inverters.dense import InverterDense, InverterProviderDenseNative
from stark.monitor import MonitorInverter
from stark.methods.resolvents import ResolventNewton
from stark.methods.schemes import SchemeKvaerno4


ROBERTSON_LAYOUT = Frame({"y": {"translation": "dy", "shape": (3,)}})

Array = Any


@DerivativeStyle.kernel(state=("y",), translation=("dy",))
def robertson_rhs(y: Array, dy: Array) -> None:
    y1 = y[0]
    y2 = y[1]
    y3 = y[2]
    coupling = 1.0e4 * y2 * y3
    quadratic = 3.0e7 * y2 * y2
    dy[0] = -0.04 * y1 + coupling
    dy[1] = 0.04 * y1 - coupling - quadratic
    dy[2] = quadratic


def robertson_jacobian_apply(state_y: Array, source_dy: Array, out_dy: Array) -> None:
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


class RobertsonLinearizer:
    __slots__ = ("apply_kernel", "dense_kernel")

    def __init__(self, accelerator) -> None:
        self.apply_kernel = accelerator.compile(
            robertson_jacobian_apply,
            label="competition-robertson-jacobian-apply",
        )
        self.dense_kernel = accelerator.compile(
            robertson_jacobian_dense,
            label="competition-robertson-jacobian-dense",
        )

    def __call__(self, interval, state, out) -> None:
        del interval
        state_y = state.y

        def apply(translation, result) -> None:
            self.apply_kernel(state_y, translation.dy, result.dy)

        def dense_fill(_basis, matrix, row_offset, column_offset, stride) -> None:
            self.dense_kernel(state_y, matrix, row_offset, column_offset, stride)

        out.apply = apply
        out.dense_fill = dense_fill


class RobertsonFullCubicResolvent:
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


class RobertsonEntryOperatorInverse:
    __slots__ = ("basis", "image", "matrix")

    def __init__(self, allocator) -> None:
        self.basis = allocator.allocate_translation()
        self.image = allocator.allocate_translation()
        self.matrix = np.zeros((3, 3), dtype=np.float64)

    def __call__(self, operator, source, target) -> None:
        basis = self.basis
        image = self.image
        matrix = self.matrix

        for column in range(3):
            basis.dy[:] = 0.0
            basis.dy[column] = 1.0
            operator(basis, image)
            matrix[:, column] = image.dy

        target.dy[:] = np.linalg.solve(matrix, source.dy)


class RobertsonTranslationBasis:
    """Coordinate basis for the Robertson translation field."""

    dimension = 3

    def vector(self, index: int, output) -> object:
        output.dy[:] = 0.0
        output.dy[index] = 1.0
        return output

    def coordinate(self, index: int, translation) -> float:
        return float(translation.dy[index])

    def coordinates(self, translation, output: list[float]) -> list[float]:
        output[:] = [float(value) for value in translation.dy]
        return output

    def synthesize(self, coordinates: list[float], output) -> object:
        output.dy[:] = coordinates
        return output


class RobertsonInverterDiagnostics:
    __slots__ = ("monitor",)

    def __init__(self, monitor: MonitorInverter) -> None:
        self.monitor = monitor

    def __call__(self) -> dict[str, Any]:
        summary = self.monitor.summary()
        self.monitor.clear()
        return {
            "inverter_solve_count": summary.solve_count,
            "inverter_failure_count": summary.failure_count,
            "inverter_iteration_min": summary.iteration_min,
            "inverter_iteration_median": summary.iteration_median,
            "inverter_iteration_max": summary.iteration_max,
            "inverter_initial_residual_median": summary.initial_residual_median,
            "inverter_final_residual_median": summary.final_residual_median,
        }


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


def stark_runtime(stark_parameters):
    engine = EngineNumpy(ROBERTSON_LAYOUT)
    linearizer = RobertsonLinearizer(engine.accelerator)
    system = System(
        derivative=robertson_rhs,
        frame=ROBERTSON_LAYOUT,
        linearizer=linearizer,
    )
    return system, engine, linearizer, stark_configuration(stark_parameters)


def stark_solver(
    name: str,
    problem_parameters,
    stark_parameters,
    initial_conditions,
    reference,
    resolvent_builder,
):
    system, engine, linearizer, configuration = stark_runtime(stark_parameters)
    resolvent, diagnostics = resolvent_builder(engine, linearizer, configuration)
    ivp = system.ivp(
        initial=initial_conditions,
        interval=Interval(problem_parameters["t0"], stark_parameters["step"], problem_parameters["t1"]),
        method=Method(scheme=SchemeKvaerno4, resolvent=resolvent),
        engine=lambda frame: engine,
        configuration=configuration,
    )

    def solve_once() -> dict[str, Any]:
        result = ivp.final_result()
        row = {
            "library": "STARK",
            "solver": name,
            "error": Comparison.fieldwise_rms_error(result.state, reference, ("y",)),
            "steps": result.steps,
        }
        if diagnostics is not None:
            row.update(diagnostics())
        return row

    return solve_once


def cubic_resolvent(engine, linearizer, configuration):
    del engine, linearizer, configuration
    return RobertsonFullCubicResolvent(tableau=SchemeKvaerno4.tableau), None


def newton_jacobi_resolvent(engine, linearizer, configuration):
    monitor = MonitorInverter()
    specialist = BlockSpecialist(engine.algebraist_specialist)
    inverter = InverterRelaxationJacobi(
        RobertsonEntryOperatorInverse(engine.allocator),
        damping=1.0,
        configuration=configuration,
        monitor=monitor,
        specialist=specialist,
    )
    return (
        ResolventNewton(
            engine.allocator,
            linearizer=linearizer,
            inverter=inverter,
            configuration=configuration,
            accelerator=engine.accelerator,
            specialist=specialist,
            tableau=SchemeKvaerno4.tableau,
        ),
        RobertsonInverterDiagnostics(monitor),
    )


def newton_dense_resolvent(engine, linearizer, configuration):
    monitor = MonitorInverter()
    specialist = BlockSpecialist(engine.algebraist_specialist)
    inverter = InverterDense(
        basis=BlockBasis([RobertsonTranslationBasis()]),
        provider=InverterProviderDenseNative(accelerator=engine.accelerator),
        monitor=monitor,
    )
    return (
        ResolventNewton(
            engine.allocator,
            linearizer=linearizer,
            inverter=inverter,
            configuration=configuration,
            accelerator=engine.accelerator,
            specialist=specialist,
            tableau=SchemeKvaerno4.tableau,
        ),
        RobertsonInverterDiagnostics(monitor),
    )


def prepare_kvaerno4_full_custom(problem_parameters, stark_parameters, initial_conditions, reference):
    return stark_solver(
        "Kvaerno4 Full Cubic",
        problem_parameters,
        stark_parameters,
        initial_conditions,
        reference,
        cubic_resolvent,
    )


def prepare_kvaerno4_full_newton(problem_parameters, stark_parameters, initial_conditions, reference):
    return stark_solver(
        "Kvaerno4 Full Newton Jacobi",
        problem_parameters,
        stark_parameters,
        initial_conditions,
        reference,
        newton_jacobi_resolvent,
    )


def prepare_kvaerno4_full_newton_dense(problem_parameters, stark_parameters, initial_conditions, reference):
    return stark_solver(
        "Kvaerno4 Full Newton Dense",
        problem_parameters,
        stark_parameters,
        initial_conditions,
        reference,
        newton_dense_resolvent,
    )
