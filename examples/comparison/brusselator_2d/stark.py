from __future__ import annotations

from math import sqrt

import numpy as np

from stark import Executor, Marcher, Integrator, Interval, Safety, Tolerance
from stark.accelerators import Accelerator
from stark.algebraist import Algebraist, AlgebraistField, AlgebraistLooped
from stark.schemes.explicit_adaptive import SchemeCashKarp, SchemeDormandPrince


try:
    ACCELERATOR = Accelerator.numba()
    USE_NUMBA_ACCELERATION = True
except ModuleNotFoundError:
    ACCELERATOR = Accelerator.none()
    USE_NUMBA_ACCELERATION = False

@ACCELERATOR.decorate
def _rhs_kernel(u, v, du, dv, alpha, a, b, inv_dx2):
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


ALGEBRAIST = Algebraist(
    fields=(
        AlgebraistField("du", "u", policy=AlgebraistLooped(rank=2)),
        AlgebraistField("dv", "v", policy=AlgebraistLooped(rank=2)),
    ),
    accelerator=ACCELERATOR,
    generate_norm="rms",
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
        du = self.u - reference["u"]
        dv = self.v - reference["v"]
        energy = np.dot(du.ravel(), du.ravel()) + np.dot(dv.ravel(), dv.ravel())
        return sqrt(float(energy) / self.u.size)


class BrusselatorTranslation:
    __slots__ = ("du", "dv")

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

    linear_combine = ALGEBRAIST.linear_combine
    __call__ = ALGEBRAIST.apply
    norm = ALGEBRAIST.norm


class BrusselatorWorkbench:
    __slots__ = ("grid_shape",)
    _compiled = False

    def __init__(self, problem_parameters):
        grid_size = problem_parameters["grid_size"]
        self.grid_shape = (grid_size, grid_size)
        if not self.__class__._compiled:
            probe = np.zeros(self.grid_shape, dtype=np.float64)
            ALGEBRAIST.compile_examples(probe, probe)
            self.__class__._compiled = True

    def __repr__(self) -> str:
        return f"BrusselatorWorkbench(grid_shape={self.grid_shape!r})"

    __str__ = __repr__

    def allocate_state(self):
        return BrusselatorState(
            np.zeros(self.grid_shape, dtype=np.float64),
            np.zeros(self.grid_shape, dtype=np.float64),
        )

    def copy_state(self, dst, src):
        np.copyto(dst.u, src.u)
        np.copyto(dst.v, src.v)

    def allocate_translation(self):
        return BrusselatorTranslation(
            np.zeros(self.grid_shape, dtype=np.float64),
            np.zeros(self.grid_shape, dtype=np.float64),
        )


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
    safety = Safety.fast()
    workbench = BrusselatorWorkbench(problem_parameters)
    derivative = BrusselatorDerivative(problem_parameters)
    scheme = SchemeCashKarp(derivative, workbench)
    marcher = Marcher(
        scheme,
        Executor(
            tolerance=Tolerance(atol=tolerance_parameters["atol"], rtol=tolerance_parameters["rtol"]),
            safety=safety,
            accelerator=ACCELERATOR,
        ),
    )
    integrate = Integrator(executor=Executor(safety=safety, accelerator=ACCELERATOR))

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
    safety = Safety.fast()
    workbench = BrusselatorWorkbench(problem_parameters)
    derivative = BrusselatorDerivative(problem_parameters)
    scheme = SchemeDormandPrince(derivative, workbench)
    marcher = Marcher(
        scheme,
        Executor(
            tolerance=Tolerance(atol=tolerance_parameters["atol"], rtol=tolerance_parameters["rtol"]),
            safety=safety,
            accelerator=ACCELERATOR,
        ),
    )
    integrate = Integrator(executor=Executor(safety=safety, accelerator=ACCELERATOR))

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
















