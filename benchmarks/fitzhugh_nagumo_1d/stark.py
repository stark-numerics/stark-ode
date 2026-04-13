from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from stark import (
    InverterPolicy,
    InverterBiCGStab,
    InverterFGMRES,
    InverterGMRES,
    InverterTolerance,
    Integrator,
    Interval,
    Marcher,
    ResolverAnderson,
    ResolverBroyden,
    ResolverNewton,
    ResolverPolicy,
    ResolverTolerance,
    Safety,
    Tolerance,
)
from stark.regulator import Regulator
from stark.jit import NUMBA_AVAILABLE, compile_if_you_can, jit_if_you_can
from stark.scheme_library import SchemeKvaerno3, SchemeKvaerno4


@jit_if_you_can
def _laplacian_periodic(field, out, inv_dx2):
    size = field.size
    for index in range(size):
        left = field[index - 1 if index > 0 else size - 1]
        centre = field[index]
        right = field[index + 1 if index + 1 < size else 0]
        out[index] = (left - 2.0 * centre + right) * inv_dx2


@jit_if_you_can
def _apply_kernel(origin_u, origin_v, delta_u, delta_v, result_u, result_v):
    result_u[:] = origin_u + delta_u
    result_v[:] = origin_v + delta_v


@jit_if_you_can
def _norm_kernel(delta_u, delta_v):
    total = 0.0
    size = delta_u.size + delta_v.size
    for value in delta_u:
        total += value * value
    for value in delta_v:
        total += value * value
    return (total / size) ** 0.5


@jit_if_you_can
def _scale_kernel(out_u, out_v, a, x_u, x_v):
    out_u[:] = a * x_u
    out_v[:] = a * x_v


@jit_if_you_can
def _combine2_kernel(out_u, out_v, a0, x0_u, x0_v, a1, x1_u, x1_v):
    out_u[:] = a0 * x0_u + a1 * x1_u
    out_v[:] = a0 * x0_v + a1 * x1_v


@jit_if_you_can
def _combine3_kernel(out_u, out_v, a0, x0_u, x0_v, a1, x1_u, x1_v, a2, x2_u, x2_v):
    out_u[:] = a0 * x0_u + a1 * x1_u + a2 * x2_u
    out_v[:] = a0 * x0_v + a1 * x1_v + a2 * x2_v


@jit_if_you_can
def _rhs_kernel(u, v, laplacian_u, out_u, out_v, diffusivity_u, epsilon, a, b):
    out_u[:] = diffusivity_u * laplacian_u + u - (u * u * u) / 3.0 - v
    out_v[:] = epsilon * (u + a - b * v)


@jit_if_you_can
def _jacobian_apply_kernel(
    u,
    translation_u,
    translation_v,
    laplacian_u,
    result_u,
    result_v,
    diffusivity_u,
    epsilon,
    b,
):
    result_u[:] = diffusivity_u * laplacian_u + (1.0 - u * u) * translation_u - translation_v
    result_v[:] = epsilon * translation_u - epsilon * b * translation_v


def _apply_translation(origin, delta, result):
    _apply_kernel(origin.u, origin.v, delta.du, delta.dv, result.u, result.v)


def _norm_translation(delta):
    return float(_norm_kernel(delta.du, delta.dv))


def _scale_translation(out, a, x):
    _scale_kernel(out.du, out.dv, a, x.du, x.dv)
    return out


def _combine2_translation(out, a0, x0, a1, x1):
    _combine2_kernel(out.du, out.dv, a0, x0.du, x0.dv, a1, x1.du, x1.dv)
    return out


def _combine3_translation(out, a0, x0, a1, x1, a2, x2):
    _combine3_kernel(out.du, out.dv, a0, x0.du, x0.dv, a1, x1.du, x1.dv, a2, x2.du, x2.dv)
    return out


@dataclass(slots=True)
class FitzHughNagumoParameters:
    grid_size: int = 128
    length: float = 40.0
    diffusivity_u: float = 1.0
    epsilon: float = 0.08
    a: float = 0.7
    b: float = 0.8
    t_start: float = 0.0
    t_stop: float = 18.0
    initial_step: float = 5.0e-3
    tolerance_atol: float = 1.0e-6
    tolerance_rtol: float = 1.0e-5
    resolution_atol: float = 1.0e-7
    resolution_rtol: float = 1.0e-7
    inversion_atol: float = 1.0e-7
    inversion_rtol: float = 1.0e-7
    resolution_max_iterations: int = 24
    inversion_max_iterations: int = 24
    inversion_restart: int = 12
    checkpoint_count: int = 36

    @property
    def dx(self) -> float:
        return self.length / self.grid_size

    @property
    def inv_dx2(self) -> float:
        return 1.0 / (self.dx * self.dx)


def parameters_from_benchmark(problem_parameters, stark_parameters, checkpoint_count):
    return FitzHughNagumoParameters(
        grid_size=problem_parameters["grid_size"],
        length=problem_parameters["length"],
        diffusivity_u=problem_parameters["diffusivity_u"],
        epsilon=problem_parameters["epsilon"],
        a=problem_parameters["a"],
        b=problem_parameters["b"],
        t_start=problem_parameters["t0"],
        t_stop=problem_parameters["t1"],
        initial_step=stark_parameters["step"],
        tolerance_atol=stark_parameters["tolerance_atol"],
        tolerance_rtol=stark_parameters["tolerance_rtol"],
        resolution_atol=stark_parameters["resolution_atol"],
        resolution_rtol=stark_parameters["resolution_rtol"],
        inversion_atol=stark_parameters["inversion_atol"],
        inversion_rtol=stark_parameters["inversion_rtol"],
        resolution_max_iterations=stark_parameters["resolution_max_iterations"],
        inversion_max_iterations=stark_parameters["inversion_max_iterations"],
        inversion_restart=stark_parameters["inversion_restart"],
        checkpoint_count=checkpoint_count,
    )


@dataclass(slots=True)
class FitzHughNagumoState:
    u: np.ndarray
    v: np.ndarray

    def __repr__(self) -> str:
        return f"FitzHughNagumoState(size={self.u.size!r})"

    __str__ = __repr__


class FitzHughNagumoTranslation:
    __slots__ = ("du", "dv")

    def __init__(self, du: np.ndarray, dv: np.ndarray) -> None:
        self.du = du
        self.dv = dv

    def __repr__(self) -> str:
        return f"FitzHughNagumoTranslation(size={self.du.size!r})"

    __str__ = __repr__

    def __call__(self, origin: FitzHughNagumoState, result: FitzHughNagumoState) -> None:
        _apply_translation(origin, self, result)

    def norm(self) -> float:
        return _norm_translation(self)

    def __add__(self, other: FitzHughNagumoTranslation) -> FitzHughNagumoTranslation:
        return FitzHughNagumoTranslation(self.du + other.du, self.dv + other.dv)

    def __rmul__(self, scalar: float) -> FitzHughNagumoTranslation:
        return FitzHughNagumoTranslation(scalar * self.du, scalar * self.dv)

    linear_combine = [_scale_translation, _combine2_translation, _combine3_translation]


class FitzHughNagumoWorkbench:
    __slots__ = ("grid_size",)
    _compiled = False

    def __init__(self, grid_size: int) -> None:
        self.grid_size = grid_size
        if not self.__class__._compiled:
            probe = np.zeros(grid_size, dtype=np.float64)
            compile_if_you_can(_laplacian_periodic, (probe, probe, 1.0))
            compile_if_you_can(_apply_kernel, (probe, probe, probe, probe, probe, probe))
            compile_if_you_can(_norm_kernel, (probe, probe))
            compile_if_you_can(_scale_kernel, (probe, probe, 1.0, probe, probe))
            compile_if_you_can(_combine2_kernel, (probe, probe, 1.0, probe, probe, 1.0, probe, probe))
            compile_if_you_can(
                _combine3_kernel,
                (probe, probe, 1.0, probe, probe, 1.0, probe, probe, 1.0, probe, probe),
            )
            self.__class__._compiled = True

    def __repr__(self) -> str:
        return f"FitzHughNagumoWorkbench(grid_size={self.grid_size!r})"

    __str__ = __repr__

    def allocate_state(self) -> FitzHughNagumoState:
        return FitzHughNagumoState(
            np.zeros(self.grid_size, dtype=np.float64),
            np.zeros(self.grid_size, dtype=np.float64),
        )

    def copy_state(self, dst: FitzHughNagumoState, src: FitzHughNagumoState) -> None:
        np.copyto(dst.u, src.u)
        np.copyto(dst.v, src.v)

    def allocate_translation(self) -> FitzHughNagumoTranslation:
        return FitzHughNagumoTranslation(
            np.zeros(self.grid_size, dtype=np.float64),
            np.zeros(self.grid_size, dtype=np.float64),
        )


class FitzHughNagumoDerivative:
    __slots__ = ("parameters", "laplacian_u")
    _compiled = False

    def __init__(self, parameters: FitzHughNagumoParameters) -> None:
        self.parameters = parameters
        self.laplacian_u = np.zeros(parameters.grid_size, dtype=np.float64)
        if not self.__class__._compiled:
            probe = np.zeros(parameters.grid_size, dtype=np.float64)
            compile_if_you_can(
                _rhs_kernel,
                (
                    probe,
                    probe,
                    probe,
                    probe,
                    probe,
                    parameters.diffusivity_u,
                    parameters.epsilon,
                    parameters.a,
                    parameters.b,
                ),
            )
            self.__class__._compiled = True

    def __repr__(self) -> str:
        return "FitzHughNagumoDerivative()"

    __str__ = __repr__

    def __call__(self, state: FitzHughNagumoState, out: FitzHughNagumoTranslation) -> None:
        parameters = self.parameters
        _laplacian_periodic(state.u, self.laplacian_u, parameters.inv_dx2)
        _rhs_kernel(
            state.u,
            state.v,
            self.laplacian_u,
            out.du,
            out.dv,
            parameters.diffusivity_u,
            parameters.epsilon,
            parameters.a,
            parameters.b,
        )


class FitzHughNagumoLinearizer:
    __slots__ = ("parameters", "operator")
    _compiled = False

    def __init__(self, parameters: FitzHughNagumoParameters) -> None:
        self.parameters = parameters
        self.operator = FitzHughNagumoJacobianOperator(parameters)
        if not self.__class__._compiled:
            probe = np.zeros(parameters.grid_size, dtype=np.float64)
            compile_if_you_can(
                _jacobian_apply_kernel,
                (
                    probe,
                    probe,
                    probe,
                    probe,
                    probe,
                    probe,
                    parameters.diffusivity_u,
                    parameters.epsilon,
                    parameters.b,
                ),
            )
            self.__class__._compiled = True

    def __repr__(self) -> str:
        return "FitzHughNagumoLinearizer()"

    __str__ = __repr__

    def __call__(self, out, state: FitzHughNagumoState) -> None:
        self.operator.configure(state)
        out.apply = self.operator


class FitzHughNagumoJacobianOperator:
    __slots__ = ("parameters", "laplacian_u", "u")

    def __init__(self, parameters: FitzHughNagumoParameters) -> None:
        self.parameters = parameters
        self.laplacian_u = np.zeros(parameters.grid_size, dtype=np.float64)
        self.u: np.ndarray | None = None

    def configure(self, state: FitzHughNagumoState) -> None:
        self.u = state.u

    def __call__(self, result: FitzHughNagumoTranslation, translation: FitzHughNagumoTranslation) -> None:
        parameters = self.parameters
        u = self.u
        assert u is not None
        _laplacian_periodic(translation.du, self.laplacian_u, parameters.inv_dx2)
        _jacobian_apply_kernel(
            u,
            translation.du,
            translation.dv,
            self.laplacian_u,
            result.du,
            result.dv,
            parameters.diffusivity_u,
            parameters.epsilon,
            parameters.b,
        )


def fitzhugh_nagumo_inner_product(left: FitzHughNagumoTranslation, right: FitzHughNagumoTranslation) -> float:
    return float(np.dot(left.du, right.du) + np.dot(left.dv, right.dv))


def initial_state(parameters: FitzHughNagumoParameters) -> tuple[np.ndarray, FitzHughNagumoState]:
    x = np.linspace(0.0, parameters.length, parameters.grid_size, endpoint=False)
    u = -1.2 + 2.4 * np.exp(-((x - 0.3 * parameters.length) ** 2) / 1.5)
    v = -0.62 + 0.1 * np.exp(-((x - 0.3 * parameters.length) ** 2) / 1.5)
    return x, FitzHughNagumoState(u.astype(np.float64), v.astype(np.float64))


class CountingMarcher:
    __slots__ = ("marcher", "steps")

    def __init__(self, marcher: Marcher) -> None:
        self.marcher = marcher
        self.steps = 0

    def __call__(self, interval, state) -> float:
        self.steps += 1
        return self.marcher(interval, state)

    def snapshot_state(self, state):
        return self.marcher.snapshot_state(state)

    def set_safety(self, safety: Safety) -> None:
        self.marcher.set_safety(safety)

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.marcher.set_apply_delta_safety(enabled)


@dataclass(slots=True)
class FitzHughNagumoTrajectory:
    label: str
    x: np.ndarray
    times: list[float]
    u_snapshots: list[np.ndarray]
    v_snapshots: list[np.ndarray]
    warmup_runtime: float
    runtime: float
    steps: int

    def plot(self, heatmap_axis, profile_axis) -> None:
        image = np.stack(self.u_snapshots, axis=0)
        extent = (float(self.x[0]), float(self.x[-1]), float(self.times[-1]), float(self.times[0]))
        heatmap_axis.imshow(image, aspect="auto", extent=extent, cmap="magma")
        heatmap_axis.set_title(f"{self.label} activator")
        heatmap_axis.set_ylabel("time")

        profile_axis.plot(self.x, self.u_snapshots[-1], label="u", color="tab:red")
        profile_axis.plot(self.x, self.v_snapshots[-1], label="v", color="tab:blue")
        profile_axis.set_title(f"{self.label} final profile")
        profile_axis.legend(loc="upper right")


def _error_against_reference(state: FitzHughNagumoState, reference) -> float:
    du = state.u - reference["u"]
    dv = state.v - reference["v"]
    return float(np.sqrt((np.dot(du, du) + np.dot(dv, dv)) / (du.size + dv.size)))


def _scheme_regulator(scheme_type):
    if scheme_type is SchemeKvaerno3:
        return Regulator(safety=0.95, error_exponent=1.0 / 3.0)
    if scheme_type is SchemeKvaerno4:
        return Regulator(safety=0.95, error_exponent=0.25)
    return Regulator(safety=0.95, error_exponent=0.45)


def _build_newton_solver(label, scheme_type, inverter_type, parameters: FitzHughNagumoParameters):
    safety = Safety.fast()
    workbench = FitzHughNagumoWorkbench(parameters.grid_size)
    derivative = FitzHughNagumoDerivative(parameters)
    linearizer = FitzHughNagumoLinearizer(parameters)
    inverter = inverter_type(
        workbench,
        fitzhugh_nagumo_inner_product,
        tolerance=InverterTolerance(
            atol=parameters.inversion_atol,
            rtol=parameters.inversion_rtol,
        ),
        policy=InverterPolicy(
            max_iterations=parameters.inversion_max_iterations,
            restart=parameters.inversion_restart,
        ),
        safety=safety,
    )
    resolver = ResolverNewton(
        workbench,
        inverter=inverter,
        tolerance=ResolverTolerance(
            atol=parameters.resolution_atol,
            rtol=parameters.resolution_rtol,
        ),
        policy=ResolverPolicy(max_iterations=parameters.resolution_max_iterations),
        safety=safety,
    )
    scheme = scheme_type(
        derivative,
        workbench,
        linearizer=linearizer,
        resolver=resolver,
        regulator=_scheme_regulator(scheme_type),
    )
    marcher = CountingMarcher(
        Marcher(
            scheme,
            tolerance=Tolerance(
                atol=parameters.tolerance_atol,
                rtol=parameters.tolerance_rtol,
            ),
            safety=safety,
        )
    )
    integrator = Integrator(safety=safety)
    return label, integrator, marcher


def _build_anderson_solver(label, scheme_type, parameters: FitzHughNagumoParameters):
    safety = Safety.fast()
    workbench = FitzHughNagumoWorkbench(parameters.grid_size)
    derivative = FitzHughNagumoDerivative(parameters)
    resolver = ResolverAnderson(
        workbench,
        fitzhugh_nagumo_inner_product,
        tolerance=ResolverTolerance(
            atol=parameters.resolution_atol,
            rtol=parameters.resolution_rtol,
        ),
        policy=ResolverPolicy(max_iterations=parameters.resolution_max_iterations),
        depth=4,
        safety=safety,
    )
    scheme = scheme_type(
        derivative,
        workbench,
        linearizer=None,
        resolver=resolver,
        regulator=_scheme_regulator(scheme_type),
    )
    marcher = CountingMarcher(
        Marcher(
            scheme,
            tolerance=Tolerance(
                atol=parameters.tolerance_atol,
                rtol=parameters.tolerance_rtol,
            ),
            safety=safety,
        )
    )
    integrator = Integrator(safety=safety)
    return label, integrator, marcher


def _build_broyden_solver(label, scheme_type, parameters: FitzHughNagumoParameters):
    safety = Safety.fast()
    workbench = FitzHughNagumoWorkbench(parameters.grid_size)
    derivative = FitzHughNagumoDerivative(parameters)
    resolver = ResolverBroyden(
        workbench,
        fitzhugh_nagumo_inner_product,
        tolerance=ResolverTolerance(
            atol=parameters.resolution_atol,
            rtol=parameters.resolution_rtol,
        ),
        policy=ResolverPolicy(max_iterations=parameters.resolution_max_iterations),
        depth=8,
        safety=safety,
    )
    scheme = scheme_type(
        derivative,
        workbench,
        linearizer=None,
        resolver=resolver,
        regulator=_scheme_regulator(scheme_type),
    )
    marcher = CountingMarcher(
        Marcher(
            scheme,
            tolerance=Tolerance(
                atol=parameters.tolerance_atol,
                rtol=parameters.tolerance_rtol,
            ),
            safety=safety,
        )
    )
    integrator = Integrator(safety=safety)
    return label, integrator, marcher


def _solve_once(label, scheme_type, inverter_type, parameters: FitzHughNagumoParameters, collect_snapshots: bool) -> FitzHughNagumoTrajectory:
    label, integrator, marcher = _build_newton_solver(label, scheme_type, inverter_type, parameters)
    x, state = initial_state(parameters)
    interval = Interval(parameters.t_start, parameters.initial_step, parameters.t_stop)
    times = [parameters.t_start]
    u_snapshots = [state.u.copy()]
    v_snapshots = [state.v.copy()]

    started = perf_counter()
    if collect_snapshots:
        for snapshot_interval, snapshot_state in integrator(
            marcher,
            interval,
            state,
            checkpoints=parameters.checkpoint_count,
        ):
            times.append(float(snapshot_interval.present))
            u_snapshots.append(snapshot_state.u.copy())
            v_snapshots.append(snapshot_state.v.copy())
    else:
        for _interval, _state in integrator.live(marcher, interval, state):
            pass
        times = [parameters.t_start, parameters.t_stop]
        u_snapshots = [state.u.copy(), state.u.copy()]
        v_snapshots = [state.v.copy(), state.v.copy()]
    runtime = perf_counter() - started

    return FitzHughNagumoTrajectory(
        label=label,
        x=x,
        times=times,
        u_snapshots=u_snapshots,
        v_snapshots=v_snapshots,
        warmup_runtime=0.0,
        runtime=runtime,
        steps=marcher.steps,
    )


def run_inverter_example(label, scheme_type, inverter_type, parameters: FitzHughNagumoParameters) -> FitzHughNagumoTrajectory:
    warmup = _solve_once(label, scheme_type, inverter_type, parameters, collect_snapshots=False)
    timed = _solve_once(label, scheme_type, inverter_type, parameters, collect_snapshots=True)
    timed.warmup_runtime = warmup.runtime
    return timed


def print_timing_table(trajectories: list[FitzHughNagumoTrajectory]) -> None:
    print("FitzHugh-Nagumo 1D inverter comparison")
    print()
    print("solver   | steps | warmup   | runtime  | final u range | final v range")
    print("---------+-------+----------+----------+---------------+--------------")
    for trajectory in trajectories:
        final_u = trajectory.u_snapshots[-1]
        final_v = trajectory.v_snapshots[-1]
        print(
            f"{trajectory.label:<8} | "
            f"{trajectory.steps:>5} | "
            f"{trajectory.warmup_runtime:>7.3f}s | "
            f"{trajectory.runtime:>7.3f}s | "
            f"{final_u.min():>+6.3f}..{final_u.max():<+6.3f} | "
            f"{final_v.min():>+6.3f}..{final_v.max():<+6.3f}"
        )
    print()
    print("Each solver is warmed once before the timed run.")
    print()
    if NUMBA_AVAILABLE:
        print("Acceleration: Numba-jitted kernels are active.")
    else:
        print("Acceleration: Python kernels are active without runtime dispatch in hot paths.")


def prepare_inverter(name, scheme_type, inverter_type, problem_parameters, stark_parameters, initial_conditions, reference):
    parameters = parameters_from_benchmark(problem_parameters, stark_parameters, checkpoint_count=2)
    _label, integrator, marcher = _build_newton_solver(name, scheme_type, inverter_type, parameters)

    def solve_once():
        state = FitzHughNagumoState(initial_conditions["u"].copy(), initial_conditions["v"].copy())
        interval = Interval(parameters.t_start, parameters.initial_step, parameters.t_stop)
        marcher.steps = 0
        for _interval, _state in integrator.live(marcher, interval, state):
            pass
        return {
            "library": "STARK",
            "solver": name,
            "error": _error_against_reference(state, reference),
            "steps": marcher.steps,
        }

    return solve_once


def prepare_newton(name, scheme_type, inverter_type, problem_parameters, stark_parameters, initial_conditions, reference):
    return prepare_inverter(name, scheme_type, inverter_type, problem_parameters, stark_parameters, initial_conditions, reference)


def prepare_quasi_newton(name, scheme_type, builder, problem_parameters, stark_parameters, initial_conditions, reference):
    parameters = parameters_from_benchmark(problem_parameters, stark_parameters, checkpoint_count=2)
    _label, integrator, marcher = builder(name, scheme_type, parameters)

    def solve_once():
        state = FitzHughNagumoState(initial_conditions["u"].copy(), initial_conditions["v"].copy())
        interval = Interval(parameters.t_start, parameters.initial_step, parameters.t_stop)
        marcher.steps = 0
        for _interval, _state in integrator.live(marcher, interval, state):
            pass
        return {
            "library": "STARK",
            "solver": name,
            "error": _error_against_reference(state, reference),
            "steps": marcher.steps,
        }

    return solve_once


def prepare_kvaerno3_gmres(problem_parameters, stark_parameters, initial_conditions, reference):
    return prepare_newton("Kvaerno3 GMRES", SchemeKvaerno3, InverterGMRES, problem_parameters, stark_parameters, initial_conditions, reference)


def prepare_kvaerno3_fgmres(problem_parameters, stark_parameters, initial_conditions, reference):
    return prepare_newton("Kvaerno3 FGMRES", SchemeKvaerno3, InverterFGMRES, problem_parameters, stark_parameters, initial_conditions, reference)


def prepare_kvaerno3_bicgstab(problem_parameters, stark_parameters, initial_conditions, reference):
    return prepare_newton("Kvaerno3 BiCGStab", SchemeKvaerno3, InverterBiCGStab, problem_parameters, stark_parameters, initial_conditions, reference)


def prepare_kvaerno3_anderson(problem_parameters, stark_parameters, initial_conditions, reference):
    return prepare_quasi_newton("Kvaerno3 Anderson", SchemeKvaerno3, _build_anderson_solver, problem_parameters, stark_parameters, initial_conditions, reference)


def prepare_kvaerno3_broyden(problem_parameters, stark_parameters, initial_conditions, reference):
    return prepare_quasi_newton("Kvaerno3 Broyden", SchemeKvaerno3, _build_broyden_solver, problem_parameters, stark_parameters, initial_conditions, reference)


def prepare_kvaerno4_gmres(problem_parameters, stark_parameters, initial_conditions, reference):
    return prepare_newton("Kvaerno4 GMRES", SchemeKvaerno4, InverterGMRES, problem_parameters, stark_parameters, initial_conditions, reference)


def prepare_kvaerno4_fgmres(problem_parameters, stark_parameters, initial_conditions, reference):
    return prepare_newton("Kvaerno4 FGMRES", SchemeKvaerno4, InverterFGMRES, problem_parameters, stark_parameters, initial_conditions, reference)


def prepare_kvaerno4_bicgstab(problem_parameters, stark_parameters, initial_conditions, reference):
    return prepare_newton("Kvaerno4 BiCGStab", SchemeKvaerno4, InverterBiCGStab, problem_parameters, stark_parameters, initial_conditions, reference)


def prepare_kvaerno4_anderson(problem_parameters, stark_parameters, initial_conditions, reference):
    return prepare_quasi_newton("Kvaerno4 Anderson", SchemeKvaerno4, _build_anderson_solver, problem_parameters, stark_parameters, initial_conditions, reference)


def prepare_kvaerno4_broyden(problem_parameters, stark_parameters, initial_conditions, reference):
    return prepare_quasi_newton("Kvaerno4 Broyden", SchemeKvaerno4, _build_broyden_solver, problem_parameters, stark_parameters, initial_conditions, reference)


def plot_trajectories(trajectories: list[FitzHughNagumoTrajectory], save: str | None, show: bool) -> None:
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - example-only dependency
        raise RuntimeError("Matplotlib is required for plotting this example.") from exc

    figure, axes = plt.subplots(
        nrows=len(trajectories),
        ncols=2,
        figsize=(12, 3.2 * len(trajectories)),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (2.4, 1.0)},
    )
    if len(trajectories) == 1:
        axes = np.array([axes])

    for row, trajectory in enumerate(trajectories):
        trajectory.plot(axes[row, 0], axes[row, 1])
        axes[row, 0].set_xlabel("x")
        axes[row, 1].set_xlabel("x")

    if save is not None:
        figure.savefig(save, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(figure)

