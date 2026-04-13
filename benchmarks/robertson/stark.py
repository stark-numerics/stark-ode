from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from stark.jit import NUMBA_AVAILABLE, compile_if_you_can, jit_if_you_can
from stark import (
    InverterPolicy,
    InverterGMRES,
    InverterTolerance,
    Integrator,
    Interval,
    Marcher,
    ResolverNewton,
    ResolverPicard,
    ResolverPolicy,
    ResolverTolerance,
    Safety,
    Tolerance,
)
from stark.regulator import Regulator
from stark.scheme_library import SchemeSDIRK21
from stark.scheme_library.implicit import SchemeBackwardEuler

@jit_if_you_can
def _apply_kernel(origin_y, delta_y, result_y):
    result_y[0] = origin_y[0] + delta_y[0]
    result_y[1] = origin_y[1] + delta_y[1]
    result_y[2] = origin_y[2] + delta_y[2]


@jit_if_you_can
def _norm_kernel(delta_y):
    total = delta_y[0] * delta_y[0] + delta_y[1] * delta_y[1] + delta_y[2] * delta_y[2]
    return (total / 3.0) ** 0.5


@jit_if_you_can
def _scale_kernel(out_y, a, x_y):
    out_y[0] = a * x_y[0]
    out_y[1] = a * x_y[1]
    out_y[2] = a * x_y[2]


@jit_if_you_can
def _combine2_kernel(out_y, a0, x0_y, a1, x1_y):
    out_y[0] = a0 * x0_y[0] + a1 * x1_y[0]
    out_y[1] = a0 * x0_y[1] + a1 * x1_y[1]
    out_y[2] = a0 * x0_y[2] + a1 * x1_y[2]


@jit_if_you_can
def _combine3_kernel(out_y, a0, x0_y, a1, x1_y, a2, x2_y):
    out_y[0] = a0 * x0_y[0] + a1 * x1_y[0] + a2 * x2_y[0]
    out_y[1] = a0 * x0_y[1] + a1 * x1_y[1] + a2 * x2_y[1]
    out_y[2] = a0 * x0_y[2] + a1 * x1_y[2] + a2 * x2_y[2]


@jit_if_you_can
def _rhs_kernel(state_y, out_y):
    y1 = state_y[0]
    y2 = state_y[1]
    y3 = state_y[2]
    out_y[0] = -0.04 * y1 + 1.0e4 * y2 * y3
    out_y[1] = 0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2
    out_y[2] = 3.0e7 * y2 * y2


@jit_if_you_can
def _jacobian_apply_kernel(y2, y3, translation_y, result_y):
    dy1 = translation_y[0]
    dy2 = translation_y[1]
    dy3 = translation_y[2]
    result_y[0] = -0.04 * dy1 + 1.0e4 * y3 * dy2 + 1.0e4 * y2 * dy3
    result_y[1] = 0.04 * dy1 + (-1.0e4 * y3 - 6.0e7 * y2) * dy2 - 1.0e4 * y2 * dy3
    result_y[2] = 6.0e7 * y2 * dy2


def _scale_translation(out, a, x):
    if NUMBA_AVAILABLE:
        _scale_kernel(out.dy, a, x.dy)
    else:
        np.multiply(x.dy, a, out=out.dy)
    return out


def _combine2_translation(out, a0, x0, a1, x1):
    if NUMBA_AVAILABLE:
        _combine2_kernel(out.dy, a0, x0.dy, a1, x1.dy)
    else:
        np.multiply(x0.dy, a0, out=out.dy)
        out.dy += a1 * x1.dy
    return out


def _combine3_translation(out, a0, x0, a1, x1, a2, x2):
    if NUMBA_AVAILABLE:
        _combine3_kernel(out.dy, a0, x0.dy, a1, x1.dy, a2, x2.dy)
    else:
        np.multiply(x0.dy, a0, out=out.dy)
        out.dy += a1 * x1.dy
        out.dy += a2 * x2.dy
    return out


@dataclass(slots=True)
class RobertsonState:
    y: np.ndarray

    def __repr__(self) -> str:
        return f"RobertsonState(y={self.y!r})"

    __str__ = __repr__

    def error_against(self, reference):
        diff = self.y - reference["y"]
        return sqrt(float(np.dot(diff, diff)) / diff.size)


class RobertsonTranslation:
    __slots__ = ("dy",)

    def __init__(self, dy):
        self.dy = dy

    def __repr__(self) -> str:
        return f"RobertsonTranslation(dy={self.dy!r})"

    __str__ = __repr__

    def __call__(self, origin, result):
        if NUMBA_AVAILABLE:
            _apply_kernel(origin.y, self.dy, result.y)
        else:
            np.add(origin.y, self.dy, out=result.y)

    def norm(self):
        if NUMBA_AVAILABLE:
            return float(_norm_kernel(self.dy))
        return sqrt(float(np.dot(self.dy, self.dy)) / self.dy.size)

    def __add__(self, other):
        return RobertsonTranslation(self.dy + other.dy)

    def __rmul__(self, scalar):
        return RobertsonTranslation(scalar * self.dy)

    linear_combine = [
        _scale_translation,
        _combine2_translation,
        _combine3_translation,
    ]


class RobertsonWorkbench:
    __slots__ = ()
    _compiled = False

    def __init__(self) -> None:
        if not self.__class__._compiled:
            probe = np.zeros(3, dtype=np.float64)
            compile_if_you_can(_apply_kernel, (probe, probe, probe))
            compile_if_you_can(_norm_kernel, (probe,))
            compile_if_you_can(_scale_kernel, (probe, 1.0, probe))
            compile_if_you_can(_combine2_kernel, (probe, 1.0, probe, 1.0, probe))
            compile_if_you_can(_combine3_kernel, (probe, 1.0, probe, 1.0, probe, 1.0, probe))
            self.__class__._compiled = True

    def __repr__(self) -> str:
        return "RobertsonWorkbench()"

    __str__ = __repr__

    def allocate_state(self):
        return RobertsonState(np.zeros(3, dtype=np.float64))

    def copy_state(self, dst, src):
        np.copyto(dst.y, src.y)

    def allocate_translation(self):
        return RobertsonTranslation(np.zeros(3, dtype=np.float64))


class RobertsonDerivative:
    __slots__ = ()
    _compiled = False

    def __init__(self) -> None:
        if not self.__class__._compiled:
            probe = np.zeros(3, dtype=np.float64)
            compile_if_you_can(_rhs_kernel, (probe, probe))
            self.__class__._compiled = True

    def __repr__(self) -> str:
        return "RobertsonDerivative()"

    __str__ = __repr__

    def __call__(self, state, out):
        if NUMBA_AVAILABLE:
            _rhs_kernel(state.y, out.dy)
        else:
            y1, y2, y3 = state.y
            out.dy[0] = -0.04 * y1 + 1.0e4 * y2 * y3
            out.dy[1] = 0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2
            out.dy[2] = 3.0e7 * y2 * y2


class RobertsonLinearizer:
    __slots__ = ()
    _compiled = False

    def __init__(self) -> None:
        if not self.__class__._compiled:
            probe = np.zeros(3, dtype=np.float64)
            compile_if_you_can(_jacobian_apply_kernel, (0.0, 0.0, probe, probe))
            self.__class__._compiled = True

    def __repr__(self) -> str:
        return "RobertsonLinearizer()"

    __str__ = __repr__

    def __call__(self, out, state):
        y2 = float(state.y[1])
        y3 = float(state.y[2])

        def apply(result, translation):
            if NUMBA_AVAILABLE:
                _jacobian_apply_kernel(y2, y3, translation.dy, result.dy)
            else:
                dy1, dy2, dy3 = translation.dy
                result.dy[0] = -0.04 * dy1 + 1.0e4 * y3 * dy2 + 1.0e4 * y2 * dy3
                result.dy[1] = 0.04 * dy1 + (-1.0e4 * y3 - 6.0e7 * y2) * dy2 - 1.0e4 * y2 * dy3
                result.dy[2] = 6.0e7 * y2 * dy2

        out.apply = apply


def robertson_inner_product(left, right):
    return float(np.dot(left.dy, right.dy))


def _initial_state(initial_conditions):
    return RobertsonState(initial_conditions["y"].copy())


def prepare_be_picard(problem_parameters, stark_parameters, initial_conditions, reference):
    safety = Safety.fast()
    workbench = RobertsonWorkbench()
    derivative = RobertsonDerivative()
    resolver = ResolverPicard(
        workbench,
        tolerance=ResolverTolerance(
            atol=stark_parameters["resolution_atol"],
            rtol=stark_parameters["resolution_rtol"],
        ),
        policy=ResolverPolicy(max_iterations=stark_parameters["resolution_max_iterations"]),
        safety=safety,
    )
    scheme = SchemeBackwardEuler(derivative, workbench, resolver=resolver)
    marcher = Marcher(scheme, tolerance=Tolerance(), safety=safety)
    integrate = Integrator(safety=safety)

    def solve_once():
        interval = Interval(problem_parameters["t0"], stark_parameters["step"], problem_parameters["t1"])
        state = _initial_state(initial_conditions)
        steps = 0

        for _interval, _state in integrate.live(marcher, interval, state):
            steps += 1

        return {
            "library": "STARK",
            "solver": "BE Picard",
            "error": state.error_against(reference),
            "steps": steps,
        }

    return solve_once


def run_be_picard(problem_parameters, stark_parameters, initial_conditions, reference):
    return prepare_be_picard(problem_parameters, stark_parameters, initial_conditions, reference)()


def prepare_be_newton(problem_parameters, stark_parameters, initial_conditions, reference):
    safety = Safety.fast()
    workbench = RobertsonWorkbench()
    derivative = RobertsonDerivative()
    linearizer = RobertsonLinearizer()
    inverter = InverterGMRES(
        workbench,
        robertson_inner_product,
        tolerance=InverterTolerance(
            atol=stark_parameters["inversion_atol"],
            rtol=stark_parameters["inversion_rtol"],
        ),
        policy=InverterPolicy(
            max_iterations=stark_parameters["inversion_max_iterations"],
            restart=stark_parameters["inversion_restart"],
        ),
        safety=safety,
    )
    resolver = ResolverNewton(
        workbench,
        inverter=inverter,
        tolerance=ResolverTolerance(
            atol=stark_parameters["resolution_atol"],
            rtol=stark_parameters["resolution_rtol"],
        ),
        policy=ResolverPolicy(max_iterations=stark_parameters["resolution_max_iterations"]),
        safety=safety,
    )
    scheme = SchemeBackwardEuler(derivative, workbench, linearizer=linearizer, resolver=resolver)
    marcher = Marcher(scheme, tolerance=Tolerance(), safety=safety)
    integrate = Integrator(safety=safety)

    def solve_once():
        interval = Interval(problem_parameters["t0"], stark_parameters["step"], problem_parameters["t1"])
        state = _initial_state(initial_conditions)
        steps = 0

        for _interval, _state in integrate.live(marcher, interval, state):
            steps += 1

        return {
            "library": "STARK",
            "solver": "BE Newton",
            "error": state.error_against(reference),
            "steps": steps,
        }

    return solve_once


def run_be_newton(problem_parameters, stark_parameters, initial_conditions, reference):
    return prepare_be_newton(problem_parameters, stark_parameters, initial_conditions, reference)()


def prepare_sdirk21_newton(problem_parameters, stark_parameters, initial_conditions, reference):
    safety = Safety.fast()
    workbench = RobertsonWorkbench()
    derivative = RobertsonDerivative()
    linearizer = RobertsonLinearizer()
    inverter = InverterGMRES(
        workbench,
        robertson_inner_product,
        tolerance=InverterTolerance(
            atol=stark_parameters["inversion_atol"],
            rtol=stark_parameters["inversion_rtol"],
        ),
        policy=InverterPolicy(
            max_iterations=stark_parameters["inversion_max_iterations"],
            restart=stark_parameters["inversion_restart"],
        ),
        safety=safety,
    )
    resolver = ResolverNewton(
        workbench,
        inverter=inverter,
        tolerance=ResolverTolerance(
            atol=stark_parameters["resolution_atol"],
            rtol=stark_parameters["resolution_rtol"],
        ),
        policy=ResolverPolicy(max_iterations=stark_parameters["resolution_max_iterations"]),
        safety=safety,
    )
    scheme = SchemeSDIRK21(
        derivative,
        workbench,
        linearizer=linearizer,
        resolver=resolver,
        regulator=Regulator(
            safety=stark_parameters.get("sdirk_regulator_safety", 1.0),
            error_exponent=stark_parameters.get("sdirk_regulator_error_exponent", 0.4),
        ),
    )
    marcher = Marcher(
        scheme,
        tolerance=Tolerance(
            atol=stark_parameters["tolerance_atol"],
            rtol=stark_parameters["tolerance_rtol"],
        ),
        safety=safety,
    )
    integrate = Integrator(safety=safety)

    def solve_once():
        interval = Interval(
            problem_parameters["t0"],
            stark_parameters.get("sdirk_step", stark_parameters["step"]),
            problem_parameters["t1"],
        )
        state = _initial_state(initial_conditions)
        steps = 0

        for _interval, _state in integrate.live(marcher, interval, state):
            steps += 1

        return {
            "library": "STARK",
            "solver": "SDIRK21 Newton",
            "error": state.error_against(reference),
            "steps": steps,
        }

    return solve_once


def run_sdirk21_newton(problem_parameters, stark_parameters, initial_conditions, reference):
    return prepare_sdirk21_newton(problem_parameters, stark_parameters, initial_conditions, reference)()

