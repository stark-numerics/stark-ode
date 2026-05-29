from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import acos, copysign, cos, pi, sqrt
from typing import Any

import numpy as np

from stark import Executor, Integrator, Interval, Marcher, ExecutorSafety, ExecutorTolerance
from stark.accelerators import Accelerator
from stark.algebraist.arity import AlgebraistArity
from stark.algebraist.generator import AlgebraistGeneratorGeneral, AlgebraistGeneratorSpecialist
from stark.algebraist.layout import AlgebraistLayout, AlgebraistLayoutField, AlgebraistLayoutUnravel
from stark.schemes import SchemeKvaerno4


try:
    ACCELERATOR = Accelerator.numba()
    USE_NUMBA_ACCELERATION = True
except ModuleNotFoundError:
    ACCELERATOR = Accelerator.none()
    USE_NUMBA_ACCELERATION = False

Array = Any

ALGEBRAIST_LAYOUT = AlgebraistLayout(
    fields=(AlgebraistLayoutField("dy", "y", policy=AlgebraistLayoutUnravel(shape=(3,))),),
)


@ACCELERATOR.compile
def _full_rhs_kernel(state_y: Array, out_y: Array) -> None:
    y1 = state_y[0]
    y2 = state_y[1]
    y3 = state_y[2]
    coupling = 1.0e4 * y2 * y3
    quadratic = 3.0e7 * y2 * y2
    out_y[0] = -0.04 * y1 + coupling
    out_y[1] = 0.04 * y1 - coupling - quadratic
    out_y[2] = quadratic


@ACCELERATOR.compile
def _apply_kernel(dy: Array, origin_y: Array, result_y: Array) -> None:
    result_y[0] = origin_y[0] + dy[0]
    result_y[1] = origin_y[1] + dy[1]
    result_y[2] = origin_y[2] + dy[2]


@ACCELERATOR.compile
def _norm_kernel(dy: Array) -> float:
    return ((dy[0] * dy[0] + dy[1] * dy[1] + dy[2] * dy[2]) / 3.0) ** 0.5


@ACCELERATOR.compile
def _state_error_kernel(y: Array, reference_y: Array) -> float:
    dy0 = y[0] - reference_y[0]
    dy1 = y[1] - reference_y[1]
    dy2 = y[2] - reference_y[2]
    return ((dy0 * dy0 + dy1 * dy1 + dy2 * dy2) / 3.0) ** 0.5


@dataclass(slots=True)
class RobertsonState:
    y: np.ndarray

    def __repr__(self) -> str:
        return f"RobertsonState(y={self.y!r})"

    __str__ = __repr__

    def error_against(self, reference):
        return float(_state_error_kernel(self.y, reference["y"]))


class RobertsonTranslation:
    __slots__ = ("dy",)

    linear_combine: tuple[Callable[..., Any], ...] = ()

    def __init__(self, dy):
        self.dy = dy

    def __repr__(self) -> str:
        return f"RobertsonTranslation(dy={self.dy!r})"

    __str__ = __repr__

    def __add__(self, other):
        return RobertsonTranslation(self.dy + other.dy)

    def __rmul__(self, scalar):
        return RobertsonTranslation(scalar * self.dy)

    def __mul__(self, scalar):
        return self.__rmul__(scalar)

    def __call__(self, origin: RobertsonState, result: RobertsonState) -> None:
        _apply_kernel(self.dy, origin.y, result.y)

    def norm(self) -> float:
        return float(_norm_kernel(self.dy))


class RobertsonAllocator:
    __slots__ = ()

    _algebraist_installed = False
    _specialist: AlgebraistGeneratorSpecialist | None = None

    def __init__(self) -> None:
        self._install_algebraist()

    def __repr__(self) -> str:
        return "RobertsonAllocator()"

    __str__ = __repr__

    def allocate_state(self):
        return RobertsonState(np.zeros(3, dtype=np.float64))

    def copy_state(self, source, out):
        np.copyto(out.y, source.y)

    def allocate_translation(self):
        return RobertsonTranslation(np.zeros(3, dtype=np.float64))

    @property
    def specialist(self):
        specialist = self.__class__._specialist
        if specialist is None:
            raise RuntimeError("RobertsonAllocator Algebraist support was not installed.")
        return specialist

    def _install_algebraist(self) -> None:
        if self.__class__._algebraist_installed:
            return

        general = AlgebraistGeneratorGeneral(
            translation=self.allocate_translation(),
            allocator=self,
            layout=ALGEBRAIST_LAYOUT,
            accelerator=ACCELERATOR,
        )
        RobertsonTranslation.linear_combine = tuple(
            general.provide(AlgebraistArity(arity))
            for arity in range(1, 13)
        )
        self.__class__._specialist = AlgebraistGeneratorSpecialist(
            translation=self.allocate_translation(),
            allocator=self,
            layout=ALGEBRAIST_LAYOUT,
            accelerator=ACCELERATOR,
        )
        self.__class__._algebraist_installed = True


class RobertsonFullDerivative:
    __slots__ = ()
    _compiled = False

    def __init__(self) -> None:
        if not self.__class__._compiled:
            probe = np.zeros(3, dtype=np.float64)
            ACCELERATOR.compile_examples(_full_rhs_kernel, (probe, probe))
            self.__class__._compiled = True

    def __repr__(self) -> str:
        return "RobertsonFullDerivative()"

    __str__ = __repr__

    def _call_numba(self, interval, state, out):
        del interval
        _full_rhs_kernel(state.y, out.dy)

    def _call_python(self, interval, state, out):
        del interval
        y1 = state.y[0]
        y2 = state.y[1]
        y3 = state.y[2]
        coupling = 1.0e4 * y2 * y3
        quadratic = 3.0e7 * y2 * y2
        out.dy[0] = -0.04 * y1 + coupling
        out.dy[1] = 0.04 * y1 - coupling - quadratic
        out.dy[2] = quadratic

RobertsonFullDerivative.__call__ = RobertsonFullDerivative._call_numba if USE_NUMBA_ACCELERATION else RobertsonFullDerivative._call_python


def _initial_state(initial_conditions):
    return RobertsonState(initial_conditions["y"].copy())


def _full_derivative():
    return RobertsonFullDerivative()


def _build_implicit_solver(scheme_type, derivative, allocator, resolvent, stark_parameters, executor_safety):
    scheme = scheme_type(
        derivative,
        allocator,
        resolvent=resolvent,
        specialist=allocator.specialist,
    )
    marcher = Marcher(
        scheme,
        Executor(
            tolerance=ExecutorTolerance(
                atol=stark_parameters["tolerance_atol"],
                rtol=stark_parameters["tolerance_rtol"],
            ),
            safety=executor_safety,
        ),
    )
    integrate = Integrator(executor=Executor(safety=executor_safety))
    return marcher, integrate


def _prepare_solver(name, marcher, integrate, problem_parameters, stark_parameters, initial_conditions, reference):
    def solve_once():
        interval = Interval(
            problem_parameters["t0"],
            stark_parameters["step"],
            problem_parameters["t1"],
        )
        state = _initial_state(initial_conditions)
        steps = 0

        for _interval, _state in integrate.live(marcher, interval, state):
            steps += 1

        return {
            "library": "STARK",
            "solver": name,
            "error": state.error_against(reference),
            "steps": steps,
        }

    return solve_once

class RobertsonFullCubicResolvent:
    __slots__ = ("tableau", "cubic")

    def __init__(self, tableau=None) -> None:
        self.tableau = tableau
        self.cubic = RobertsonCubicRoot()

    def __repr__(self) -> str:
        return f"RobertsonFullCubicResolvent(tableau={self.tableau!r})"

    __str__ = __repr__

    def bind(self, interval, state) -> None:
        del interval, state

    def __call__(self, problem, out) -> None:
        state = problem.origin
        alpha = problem.alpha
        rhs = problem.rhs

        delta = out[0]
        if alpha == 0.0:
            if rhs is None:
                delta.dy[0] = 0.0
                delta.dy[1] = 0.0
                delta.dy[2] = 0.0
            else:
                np.copyto(delta.dy, rhs[0].dy)
            return

        y = state.y
        rhs_y = rhs[0].dy if rhs is not None else None
        rhs0 = float(rhs_y[0]) if rhs_y is not None else 0.0
        rhs1 = float(rhs_y[1]) if rhs_y is not None else 0.0
        rhs2 = float(rhs_y[2]) if rhs_y is not None else 0.0

        shifted_y2 = float(y[1]) + rhs1
        shifted_y3 = float(y[2]) + rhs2
        total = float(y[0] + y[1] + y[2] + rhs0 + rhs1 + rhs2)

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

        delta.dy[0] = z1 - y[0]
        delta.dy[1] = z2 - y[1]
        delta.dy[2] = z3 - y[2]


class RobertsonCubicRoot:
    __slots__ = ()

    @staticmethod
    def _cbrt(value: float) -> float:
        if value == 0.0:
            return 0.0
        return copysign(abs(value) ** (1.0 / 3.0), value)

    def solve(
        self,
        coefficient3: float,
        coefficient2: float,
        coefficient1: float,
        coefficient0: float,
        *,
        lower: float,
        upper: float,
        target: float,
    ) -> float:
        a = coefficient2 / coefficient3
        b = coefficient1 / coefficient3
        c = coefficient0 / coefficient3
        p = b - (a * a) / 3.0
        q = (2.0 * a * a * a) / 27.0 - (a * b) / 3.0 + c
        half_q = 0.5 * q
        third_p = p / 3.0
        discriminant = half_q * half_q + third_p * third_p * third_p

        if discriminant >= 0.0:
            root = self._cbrt(-half_q + sqrt(discriminant)) + self._cbrt(-half_q - sqrt(discriminant))
            return root - a / 3.0

        radius = 2.0 * sqrt(-third_p)
        angle = acos(max(-1.0, min(1.0, -half_q / sqrt(-(third_p * third_p * third_p)))))
        roots = [
            radius * cos((angle + 2.0 * pi * index) / 3.0) - a / 3.0
            for index in range(3)
        ]
        bounded = [root for root in roots if lower - 1.0e-12 <= root <= upper + 1.0e-12]
        if bounded:
            return min(bounded, key=lambda root: abs(root - target))
        return min(roots, key=lambda root: abs(root - target))
def prepare_implicit_custom(name, scheme_type, problem_parameters, stark_parameters, initial_conditions, reference):
    executor_safety = ExecutorSafety.fast()
    allocator = RobertsonAllocator()
    derivative = _full_derivative()
    marcher, integrate = _build_implicit_solver(
        scheme_type,
        derivative,
        allocator,
        RobertsonFullCubicResolvent(tableau=scheme_type.tableau),
        stark_parameters,
        executor_safety,
    )
    return _prepare_solver(name, marcher, integrate, problem_parameters, stark_parameters, initial_conditions, reference)
def prepare_kvaerno4_full_custom(problem_parameters, stark_parameters, initial_conditions, reference):
    return prepare_implicit_custom(
        "Kvaerno4 Full Cubic",
        SchemeKvaerno4,
        problem_parameters,
        stark_parameters,
        initial_conditions,
        reference,
    )








