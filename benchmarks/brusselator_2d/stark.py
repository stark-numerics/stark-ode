from __future__ import annotations

from math import sqrt

import numpy as np

from stark.jit import NUMBA_AVAILABLE, compile_if_you_can, jit_if_you_can
from stark import Marcher, Integrator, Interval, Safety, Tolerance
from stark.scheme_library.adaptive import SchemeCashKarp, SchemeDormandPrince

@jit_if_you_can
def _apply_kernel(origin_u, origin_v, du, dv, result_u, result_v):
    rows, cols = origin_u.shape
    for i in range(rows):
        for j in range(cols):
            result_u[i, j] = origin_u[i, j] + du[i, j]
            result_v[i, j] = origin_v[i, j] + dv[i, j]


@jit_if_you_can
def _norm_kernel(du, dv):
    rows, cols = du.shape
    total = 0.0
    for i in range(rows):
        for j in range(cols):
            total += du[i, j] * du[i, j] + dv[i, j] * dv[i, j]
    return (total / (rows * cols)) ** 0.5


@jit_if_you_can
def _scale_kernel(out_du, out_dv, a, x_du, x_dv):
    rows, cols = out_du.shape
    for i in range(rows):
        for j in range(cols):
            out_du[i, j] = a * x_du[i, j]
            out_dv[i, j] = a * x_dv[i, j]


@jit_if_you_can
def _combine2_kernel(out_du, out_dv, a0, x0_du, x0_dv, a1, x1_du, x1_dv):
    rows, cols = out_du.shape
    for i in range(rows):
        for j in range(cols):
            out_du[i, j] = a0 * x0_du[i, j] + a1 * x1_du[i, j]
            out_dv[i, j] = a0 * x0_dv[i, j] + a1 * x1_dv[i, j]


@jit_if_you_can
def _combine3_kernel(out_du, out_dv, a0, x0_du, x0_dv, a1, x1_du, x1_dv, a2, x2_du, x2_dv):
    rows, cols = out_du.shape
    for i in range(rows):
        for j in range(cols):
            out_du[i, j] = a0 * x0_du[i, j] + a1 * x1_du[i, j] + a2 * x2_du[i, j]
            out_dv[i, j] = a0 * x0_dv[i, j] + a1 * x1_dv[i, j] + a2 * x2_dv[i, j]


@jit_if_you_can
def _combine4_kernel(
    out_du,
    out_dv,
    a0,
    x0_du,
    x0_dv,
    a1,
    x1_du,
    x1_dv,
    a2,
    x2_du,
    x2_dv,
    a3,
    x3_du,
    x3_dv,
):
    rows, cols = out_du.shape
    for i in range(rows):
        for j in range(cols):
            out_du[i, j] = (
                a0 * x0_du[i, j]
                + a1 * x1_du[i, j]
                + a2 * x2_du[i, j]
                + a3 * x3_du[i, j]
            )
            out_dv[i, j] = (
                a0 * x0_dv[i, j]
                + a1 * x1_dv[i, j]
                + a2 * x2_dv[i, j]
                + a3 * x3_dv[i, j]
            )


@jit_if_you_can
def _combine5_kernel(
    out_du,
    out_dv,
    a0,
    x0_du,
    x0_dv,
    a1,
    x1_du,
    x1_dv,
    a2,
    x2_du,
    x2_dv,
    a3,
    x3_du,
    x3_dv,
    a4,
    x4_du,
    x4_dv,
):
    rows, cols = out_du.shape
    for i in range(rows):
        for j in range(cols):
            out_du[i, j] = (
                a0 * x0_du[i, j]
                + a1 * x1_du[i, j]
                + a2 * x2_du[i, j]
                + a3 * x3_du[i, j]
                + a4 * x4_du[i, j]
            )
            out_dv[i, j] = (
                a0 * x0_dv[i, j]
                + a1 * x1_dv[i, j]
                + a2 * x2_dv[i, j]
                + a3 * x3_dv[i, j]
                + a4 * x4_dv[i, j]
            )


@jit_if_you_can
def _combine6_kernel(
    out_du,
    out_dv,
    a0,
    x0_du,
    x0_dv,
    a1,
    x1_du,
    x1_dv,
    a2,
    x2_du,
    x2_dv,
    a3,
    x3_du,
    x3_dv,
    a4,
    x4_du,
    x4_dv,
    a5,
    x5_du,
    x5_dv,
):
    rows, cols = out_du.shape
    for i in range(rows):
        for j in range(cols):
            out_du[i, j] = (
                a0 * x0_du[i, j]
                + a1 * x1_du[i, j]
                + a2 * x2_du[i, j]
                + a3 * x3_du[i, j]
                + a4 * x4_du[i, j]
                + a5 * x5_du[i, j]
            )
            out_dv[i, j] = (
                a0 * x0_dv[i, j]
                + a1 * x1_dv[i, j]
                + a2 * x2_dv[i, j]
                + a3 * x3_dv[i, j]
                + a4 * x4_dv[i, j]
                + a5 * x5_dv[i, j]
            )


@jit_if_you_can
def _combine7_kernel(
    out_du,
    out_dv,
    a0,
    x0_du,
    x0_dv,
    a1,
    x1_du,
    x1_dv,
    a2,
    x2_du,
    x2_dv,
    a3,
    x3_du,
    x3_dv,
    a4,
    x4_du,
    x4_dv,
    a5,
    x5_du,
    x5_dv,
    a6,
    x6_du,
    x6_dv,
):
    rows, cols = out_du.shape
    for i in range(rows):
        for j in range(cols):
            out_du[i, j] = (
                a0 * x0_du[i, j]
                + a1 * x1_du[i, j]
                + a2 * x2_du[i, j]
                + a3 * x3_du[i, j]
                + a4 * x4_du[i, j]
                + a5 * x5_du[i, j]
                + a6 * x6_du[i, j]
            )
            out_dv[i, j] = (
                a0 * x0_dv[i, j]
                + a1 * x1_dv[i, j]
                + a2 * x2_dv[i, j]
                + a3 * x3_dv[i, j]
                + a4 * x4_dv[i, j]
                + a5 * x5_dv[i, j]
                + a6 * x6_dv[i, j]
            )


@jit_if_you_can
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


def _scale_translation(out, a, x):
    if NUMBA_AVAILABLE:
        _scale_kernel(out.du, out.dv, a, x.du, x.dv)
    else:
        np.multiply(x.du, a, out=out.du)
        np.multiply(x.dv, a, out=out.dv)
    return out


def _combine2_translation(out, a0, x0, a1, x1):
    if NUMBA_AVAILABLE:
        _combine2_kernel(out.du, out.dv, a0, x0.du, x0.dv, a1, x1.du, x1.dv)
    else:
        np.multiply(x0.du, a0, out=out.du)
        out.du += a1 * x1.du
        np.multiply(x0.dv, a0, out=out.dv)
        out.dv += a1 * x1.dv
    return out


def _combine3_translation(out, a0, x0, a1, x1, a2, x2):
    if NUMBA_AVAILABLE:
        _combine3_kernel(out.du, out.dv, a0, x0.du, x0.dv, a1, x1.du, x1.dv, a2, x2.du, x2.dv)
    else:
        np.multiply(x0.du, a0, out=out.du)
        out.du += a1 * x1.du
        out.du += a2 * x2.du
        np.multiply(x0.dv, a0, out=out.dv)
        out.dv += a1 * x1.dv
        out.dv += a2 * x2.dv
    return out


def _combine4_translation(out, a0, x0, a1, x1, a2, x2, a3, x3):
    if NUMBA_AVAILABLE:
        _combine4_kernel(
            out.du,
            out.dv,
            a0,
            x0.du,
            x0.dv,
            a1,
            x1.du,
            x1.dv,
            a2,
            x2.du,
            x2.dv,
            a3,
            x3.du,
            x3.dv,
        )
    else:
        np.multiply(x0.du, a0, out=out.du)
        out.du += a1 * x1.du
        out.du += a2 * x2.du
        out.du += a3 * x3.du
        np.multiply(x0.dv, a0, out=out.dv)
        out.dv += a1 * x1.dv
        out.dv += a2 * x2.dv
        out.dv += a3 * x3.dv
    return out


def _combine5_translation(out, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4):
    if NUMBA_AVAILABLE:
        _combine5_kernel(
            out.du,
            out.dv,
            a0,
            x0.du,
            x0.dv,
            a1,
            x1.du,
            x1.dv,
            a2,
            x2.du,
            x2.dv,
            a3,
            x3.du,
            x3.dv,
            a4,
            x4.du,
            x4.dv,
        )
    else:
        np.multiply(x0.du, a0, out=out.du)
        out.du += a1 * x1.du
        out.du += a2 * x2.du
        out.du += a3 * x3.du
        out.du += a4 * x4.du
        np.multiply(x0.dv, a0, out=out.dv)
        out.dv += a1 * x1.dv
        out.dv += a2 * x2.dv
        out.dv += a3 * x3.dv
        out.dv += a4 * x4.dv
    return out


def _combine6_translation(out, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5):
    if NUMBA_AVAILABLE:
        _combine6_kernel(
            out.du,
            out.dv,
            a0,
            x0.du,
            x0.dv,
            a1,
            x1.du,
            x1.dv,
            a2,
            x2.du,
            x2.dv,
            a3,
            x3.du,
            x3.dv,
            a4,
            x4.du,
            x4.dv,
            a5,
            x5.du,
            x5.dv,
        )
    else:
        np.multiply(x0.du, a0, out=out.du)
        out.du += a1 * x1.du
        out.du += a2 * x2.du
        out.du += a3 * x3.du
        out.du += a4 * x4.du
        out.du += a5 * x5.du
        np.multiply(x0.dv, a0, out=out.dv)
        out.dv += a1 * x1.dv
        out.dv += a2 * x2.dv
        out.dv += a3 * x3.dv
        out.dv += a4 * x4.dv
        out.dv += a5 * x5.dv
    return out


def _combine7_translation(out, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6):
    if NUMBA_AVAILABLE:
        _combine7_kernel(
            out.du,
            out.dv,
            a0,
            x0.du,
            x0.dv,
            a1,
            x1.du,
            x1.dv,
            a2,
            x2.du,
            x2.dv,
            a3,
            x3.du,
            x3.dv,
            a4,
            x4.du,
            x4.dv,
            a5,
            x5.du,
            x5.dv,
            a6,
            x6.du,
            x6.dv,
        )
    else:
        np.multiply(x0.du, a0, out=out.du)
        out.du += a1 * x1.du
        out.du += a2 * x2.du
        out.du += a3 * x3.du
        out.du += a4 * x4.du
        out.du += a5 * x5.du
        out.du += a6 * x6.du
        np.multiply(x0.dv, a0, out=out.dv)
        out.dv += a1 * x1.dv
        out.dv += a2 * x2.dv
        out.dv += a3 * x3.dv
        out.dv += a4 * x4.dv
        out.dv += a5 * x5.dv
        out.dv += a6 * x6.dv
    return out


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

    def __call__(self, origin, result):
        if NUMBA_AVAILABLE:
            _apply_kernel(origin.u, origin.v, self.du, self.dv, result.u, result.v)
        else:
            np.add(origin.u, self.du, out=result.u)
            np.add(origin.v, self.dv, out=result.v)

    def norm(self):
        if NUMBA_AVAILABLE:
            return float(_norm_kernel(self.du, self.dv))
        energy = np.dot(self.du.ravel(), self.du.ravel()) + np.dot(self.dv.ravel(), self.dv.ravel())
        return sqrt(float(energy) / self.du.size)

    def __add__(self, other):
        return BrusselatorTranslation(self.du + other.du, self.dv + other.dv)

    def __rmul__(self, scalar):
        return BrusselatorTranslation(scalar * self.du, scalar * self.dv)

    linear_combine = [
        _scale_translation,
        _combine2_translation,
        _combine3_translation,
        _combine4_translation,
        _combine5_translation,
        _combine6_translation,
        _combine7_translation,
    ]


class BrusselatorWorkbench:
    __slots__ = ("grid_shape",)
    _compiled = False

    def __init__(self, problem_parameters):
        grid_size = problem_parameters["grid_size"]
        self.grid_shape = (grid_size, grid_size)
        if not self.__class__._compiled:
            probe = np.zeros(self.grid_shape, dtype=np.float64)
            compile_if_you_can(_apply_kernel, (probe, probe, probe, probe, probe, probe))
            compile_if_you_can(_norm_kernel, (probe, probe))
            compile_if_you_can(_scale_kernel, (probe, probe, 1.0, probe, probe))
            compile_if_you_can(_combine2_kernel, (probe, probe, 1.0, probe, probe, 1.0, probe, probe))
            compile_if_you_can(
                _combine3_kernel,
                (probe, probe, 1.0, probe, probe, 1.0, probe, probe, 1.0, probe, probe),
            )
            compile_if_you_can(
                _combine4_kernel,
                (probe, probe, 1.0, probe, probe, 1.0, probe, probe, 1.0, probe, probe, 1.0, probe, probe),
            )
            compile_if_you_can(
                _combine5_kernel,
                (
                    probe,
                    probe,
                    1.0,
                    probe,
                    probe,
                    1.0,
                    probe,
                    probe,
                    1.0,
                    probe,
                    probe,
                    1.0,
                    probe,
                    probe,
                    1.0,
                    probe,
                    probe,
                ),
            )
            compile_if_you_can(
                _combine6_kernel,
                (
                    probe,
                    probe,
                    1.0,
                    probe,
                    probe,
                    1.0,
                    probe,
                    probe,
                    1.0,
                    probe,
                    probe,
                    1.0,
                    probe,
                    probe,
                    1.0,
                    probe,
                    probe,
                    1.0,
                    probe,
                    probe,
                ),
            )
            compile_if_you_can(
                _combine7_kernel,
                (
                    probe,
                    probe,
                    1.0,
                    probe,
                    probe,
                    1.0,
                    probe,
                    probe,
                    1.0,
                    probe,
                    probe,
                    1.0,
                    probe,
                    probe,
                    1.0,
                    probe,
                    probe,
                    1.0,
                    probe,
                    probe,
                    1.0,
                    probe,
                    probe,
                ),
            )
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
            compile_if_you_can(
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

    def __call__(self, state, out):
        if NUMBA_AVAILABLE:
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
        tolerance=Tolerance(atol=tolerance_parameters["atol"], rtol=tolerance_parameters["rtol"]),
        safety=safety,
    )
    integrate = Integrator(safety=safety)

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
        tolerance=Tolerance(atol=tolerance_parameters["atol"], rtol=tolerance_parameters["rtol"]),
        safety=safety,
    )
    integrate = Integrator(safety=safety)

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
