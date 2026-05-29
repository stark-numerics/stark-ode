from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from stark import Executor, Marcher, Integrator, Interval, ExecutorSafety, ExecutorTolerance
from stark.accelerators import Accelerator
from stark.algebraist.arity import AlgebraistArity
from stark.algebraist.generator import AlgebraistGeneratorGeneral, AlgebraistGeneratorSpecialist
from stark.algebraist.layout import AlgebraistLayout, AlgebraistLayoutField, AlgebraistLayoutLooped
from stark.schemes.explicit_adaptive import SchemeCashKarp, SchemeDormandPrince


try:
    ACCELERATOR = Accelerator.numba()
    USE_NUMBA_ACCELERATION = True
except ModuleNotFoundError:
    ACCELERATOR = Accelerator.none()
    USE_NUMBA_ACCELERATION = False


Array = Any


@ACCELERATOR.compile
def _rhs_kernel(u: Array, v: Array, du: Array, dv: Array, alpha: float, a: float, b: float, inv_dx2: float) -> None:
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
            lap_u = (u[im1, j] + u[ip1, j] + u[i, jm1] + u[i, jp1] - 4.0 * u_ij) * inv_dx2
            lap_v = (v[im1, j] + v[ip1, j] + v[i, jm1] + v[i, jp1] - 4.0 * v_ij) * inv_dx2
            du[i, j] = alpha * lap_u + a + reaction - (b + 1.0) * u_ij
            dv[i, j] = alpha * lap_v + b * u_ij - reaction


@ACCELERATOR.compile
def _apply_kernel(
    du: Array,
    dv: Array,
    origin_u: Array,
    origin_v: Array,
    result_u: Array,
    result_v: Array,
) -> None:
    rows, cols = du.shape
    for i in range(rows):
        for j in range(cols):
            result_u[i, j] = origin_u[i, j] + du[i, j]
            result_v[i, j] = origin_v[i, j] + dv[i, j]


@ACCELERATOR.compile
def _norm_kernel(du: Array, dv: Array) -> float:
    rows, cols = du.shape
    total = 0.0
    for i in range(rows):
        for j in range(cols):
            total += du[i, j] * du[i, j] + dv[i, j] * dv[i, j]
    return (total / du.size) ** 0.5


@ACCELERATOR.compile
def _state_error_kernel(
    u: Array,
    v: Array,
    reference_u: Array,
    reference_v: Array,
) -> float:
    rows, cols = u.shape
    total = 0.0
    for i in range(rows):
        for j in range(cols):
            u_error = u[i, j] - reference_u[i, j]
            v_error = v[i, j] - reference_v[i, j]
            total += u_error * u_error + v_error * v_error
    return (total / u.size) ** 0.5


ALGEBRAIST_LAYOUT = AlgebraistLayout(
    fields=(
        AlgebraistLayoutField("du", "u", policy=AlgebraistLayoutLooped(rank=2)),
        AlgebraistLayoutField("dv", "v", policy=AlgebraistLayoutLooped(rank=2)),
    ),
)


class BrusselatorState:
    __slots__ = ("u", "v")

    def __init__(self, u, v):
        self.u = u
        self.v = v

    def __repr__(self) -> str:
        return f"BrusselatorState(shape={self.u.shape!r})"

    __str__ = __repr__

    def error_against(self, reference):
        return float(
            _state_error_kernel(
                self.u,
                self.v,
                reference["u"],
                reference["v"],
            )
        )


class BrusselatorTranslation:
    __slots__ = ("du", "dv")

    linear_combine: tuple[Callable[..., Any], ...] = ()

    def __init__(self, du, dv):
        self.du = du
        self.dv = dv

    def __repr__(self) -> str:
        return f"BrusselatorTranslation(shape={self.du.shape!r}, norm={self.norm():.6g})"

    __str__ = __repr__

    def __add__(self, other):
        return BrusselatorTranslation(self.du + other.du, self.dv + other.dv)

    def __rmul__(self, scalar):
        return BrusselatorTranslation(scalar * self.du, scalar * self.dv)

    def __mul__(self, scalar):
        return self.__rmul__(scalar)

    def __call__(self, origin, result):
        _apply_kernel(
            self.du,
            self.dv,
            origin.u,
            origin.v,
            result.u,
            result.v,
        )

    def norm(self):
        return float(_norm_kernel(self.du, self.dv))


class BrusselatorAllocator:
    __slots__ = ("grid_shape",)

    _algebraist_installed = False
    _specialist: AlgebraistGeneratorSpecialist | None = None

    def __init__(self, problem_parameters):
        grid_size = problem_parameters["grid_size"]
        self.grid_shape = (grid_size, grid_size)
        self._install_algebraist()

    def __repr__(self) -> str:
        return f"BrusselatorAllocator(grid_shape={self.grid_shape!r})"

    __str__ = __repr__

    def allocate_state(self):
        return BrusselatorState(
            np.zeros(self.grid_shape, dtype=np.float64),
            np.zeros(self.grid_shape, dtype=np.float64),
        )

    def copy_state(self, source, out):
        np.copyto(out.u, source.u)
        np.copyto(out.v, source.v)

    def allocate_translation(self):
        return BrusselatorTranslation(
            np.zeros(self.grid_shape, dtype=np.float64),
            np.zeros(self.grid_shape, dtype=np.float64),
        )

    @property
    def specialist(self):
        specialist = self.__class__._specialist
        if specialist is None:
            raise RuntimeError("BrusselatorAllocator Algebraist support was not installed.")
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
        BrusselatorTranslation.linear_combine = tuple(
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


class BrusselatorDerivative:
    __slots__ = ("a", "alpha", "b", "inv_dx2")
    _compiled = False

    def __init__(self, problem_parameters):
        self.alpha = problem_parameters["alpha"]
        self.a = problem_parameters["a"]
        self.b = problem_parameters["b"]
        self.inv_dx2 = problem_parameters["inv_dx2"]
        if not self.__class__._compiled:
            probe = np.zeros((problem_parameters["grid_size"], problem_parameters["grid_size"]), dtype=np.float64)
            ACCELERATOR.compile_examples(
                _rhs_kernel,
                (probe, probe, probe, probe, self.alpha, self.a, self.b, self.inv_dx2),
            )
            self.__class__._compiled = True

    def __repr__(self) -> str:
        return (
            "BrusselatorDerivative("
            f"alpha={self.alpha!r}, a={self.a!r}, b={self.b!r}, inv_dx2={self.inv_dx2!r})"
        )

    __str__ = __repr__

    def __call__(self, interval, state, out):
        del interval
        if USE_NUMBA_ACCELERATION:
            _rhs_kernel(state.u, state.v, out.du, out.dv, self.alpha, self.a, self.b, self.inv_dx2)
            return

        u = state.u
        v = state.v
        lap_u = (
            np.roll(u, 1, axis=0)
            + np.roll(u, -1, axis=0)
            + np.roll(u, 1, axis=1)
            + np.roll(u, -1, axis=1)
            - 4.0 * u
        ) * self.inv_dx2
        lap_v = (
            np.roll(v, 1, axis=0)
            + np.roll(v, -1, axis=0)
            + np.roll(v, 1, axis=1)
            + np.roll(v, -1, axis=1)
            - 4.0 * v
        ) * self.inv_dx2
        reaction = u * u * v
        out.du[:] = self.alpha * lap_u + self.a + reaction - (self.b + 1.0) * u
        out.dv[:] = self.alpha * lap_v + self.b * u - reaction


def prepare_rkck(problem_parameters, tolerance_parameters, initial_conditions, reference):
    executor_safety = ExecutorSafety.fast()
    allocator = BrusselatorAllocator(problem_parameters)
    derivative = BrusselatorDerivative(problem_parameters)
    scheme = SchemeCashKarp(derivative, allocator, specialist=allocator.specialist)
    marcher = Marcher(
        scheme,
        Executor(
            tolerance=ExecutorTolerance(atol=tolerance_parameters["atol"], rtol=tolerance_parameters["rtol"]),
            safety=executor_safety,
        ),
    )
    integrate = Integrator(executor=Executor(safety=executor_safety))

    def solve_once():
        interval = Interval(
            problem_parameters["t0"],
            tolerance_parameters["initial_step"],
            problem_parameters["t1"],
        )
        state = BrusselatorState(initial_conditions["u"].copy(), initial_conditions["v"].copy())
        steps = 0

        for _interval, _state in integrate.live(marcher, interval, state):
            steps += 1

        return {
            "library": "STARK",
            "solver": "RKCK",
            "error": state.error_against(reference),
            "steps": steps,
        }

    return solve_once


def run_rkck(problem_parameters, tolerance_parameters, initial_conditions, reference):
    return prepare_rkck(problem_parameters, tolerance_parameters, initial_conditions, reference)()


def prepare_rkdp(problem_parameters, tolerance_parameters, initial_conditions, reference):
    executor_safety = ExecutorSafety.fast()
    allocator = BrusselatorAllocator(problem_parameters)
    derivative = BrusselatorDerivative(problem_parameters)
    scheme = SchemeDormandPrince(derivative, allocator, specialist=allocator.specialist)
    marcher = Marcher(
        scheme,
        Executor(
            tolerance=ExecutorTolerance(atol=tolerance_parameters["atol"], rtol=tolerance_parameters["rtol"]),
            safety=executor_safety,
        ),
    )
    integrate = Integrator(executor=Executor(safety=executor_safety))

    def solve_once():
        interval = Interval(
            problem_parameters["t0"],
            tolerance_parameters["initial_step"],
            problem_parameters["t1"],
        )
        state = BrusselatorState(initial_conditions["u"].copy(), initial_conditions["v"].copy())
        steps = 0

        for _interval, _state in integrate.live(marcher, interval, state):
            steps += 1

        return {
            "library": "STARK",
            "solver": "RKDP",
            "error": state.error_against(reference),
            "steps": steps,
        }

    return solve_once


def run_rkdp(problem_parameters, tolerance_parameters, initial_conditions, reference):
    return prepare_rkdp(problem_parameters, tolerance_parameters, initial_conditions, reference)()














