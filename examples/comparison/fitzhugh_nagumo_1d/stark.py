from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from stark import Executor, ImExDerivative, Integrator, Interval, Marcher, Safety, Tolerance
from stark.accelerators import Accelerator
from stark.algebraist import Algebraist, AlgebraistField, AlgebraistLooped
from stark.execution.regulator import Regulator
from stark.inverters import InverterPolicy, InverterTolerance
from stark.resolvents import ResolventAnderson, ResolventNewton, ResolventPolicy, ResolventTolerance
from stark.schemes import SchemeKennedyCarpenter43_7, SchemeKvaerno3


try:
    ACCELERATOR = Accelerator.numba()
    USE_NUMBA_ACCELERATION = True
except ModuleNotFoundError:
    ACCELERATOR = Accelerator.none()
    USE_NUMBA_ACCELERATION = False

ALGEBRAIST = Algebraist(
    fields=(
        AlgebraistField("du", "u", policy=AlgebraistLooped(rank=1)),
        AlgebraistField("dv", "v", policy=AlgebraistLooped(rank=1)),
    ),
    accelerator=ACCELERATOR,
    generate_norm="rms",
)


@ACCELERATOR.decorate
def _laplacian_periodic(field, out, inv_dx2):
    size = field.size
    for index in range(size):
        left = field[index - 1 if index > 0 else size - 1]
        centre = field[index]
        right = field[index + 1 if index + 1 < size else 0]
        out[index] = (left - 2.0 * centre + right) * inv_dx2


@ACCELERATOR.decorate
def _rhs_kernel(u, v, laplacian_u, out_u, out_v, diffusivity_u, epsilon, a, b):
    out_u[:] = diffusivity_u * laplacian_u + u - (u * u * u) / 3.0 - v
    out_v[:] = epsilon * (u + a - b * v)


@ACCELERATOR.decorate
def _explicit_rhs_kernel(u, v, out_u, out_v, epsilon, a, b):
    out_u[:] = u - (u * u * u) / 3.0 - v
    out_v[:] = epsilon * (u + a - b * v)


@ACCELERATOR.decorate
def _implicit_rhs_kernel(laplacian_u, out_u, out_v, diffusivity_u):
    out_u[:] = diffusivity_u * laplacian_u
    out_v[:] = 0.0

def _translation_inner_product(left, right):
    return float(np.dot(left.du, right.du) + np.dot(left.dv, right.dv))


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
    checkpoint_count: int = 100
    tolerance_atol: float = 1.0e-6
    tolerance_rtol: float = 1.0e-5
    resolution_atol: float = 1.0e-7
    resolution_rtol: float = 1.0e-7
    resolution_max_iterations: int = 24
    inversion_atol: float = 1.0e-7
    inversion_rtol: float = 1.0e-7
    inversion_max_iterations: int = 24
    inversion_restart: int = 12

    @property
    def dx(self) -> float:
        return self.length / self.grid_size

    @property
    def inv_dx2(self) -> float:
        return 1.0 / (self.dx * self.dx)


def parameters_from_benchmark(problem_parameters, stark_parameters):
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
        checkpoint_count=stark_parameters.get("checkpoints", 100),
        tolerance_atol=stark_parameters["tolerance_atol"],
        tolerance_rtol=stark_parameters["tolerance_rtol"],
        resolution_atol=stark_parameters["resolution_atol"],
        resolution_rtol=stark_parameters["resolution_rtol"],
        resolution_max_iterations=stark_parameters["resolution_max_iterations"],
        inversion_atol=stark_parameters.get("inversion_atol", 1.0e-7),
        inversion_rtol=stark_parameters.get("inversion_rtol", 1.0e-7),
        inversion_max_iterations=stark_parameters.get("inversion_max_iterations", 24),
        inversion_restart=stark_parameters.get("inversion_restart", 12),
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

    def __add__(self, other: FitzHughNagumoTranslation) -> FitzHughNagumoTranslation:
        return FitzHughNagumoTranslation(self.du + other.du, self.dv + other.dv)

    def __rmul__(self, scalar: float) -> FitzHughNagumoTranslation:
        return FitzHughNagumoTranslation(scalar * self.du, scalar * self.dv)

    linear_combine = ALGEBRAIST.linear_combine
    __call__ = ALGEBRAIST.apply
    norm = ALGEBRAIST.norm


class FitzHughNagumoWorkbench:
    __slots__ = ("grid_size",)
    _compiled = False

    def __init__(self, grid_size: int) -> None:
        self.grid_size = grid_size
        if not self.__class__._compiled:
            probe = np.zeros(grid_size, dtype=np.float64)
            ACCELERATOR.compile_examples(_laplacian_periodic, (probe, probe, 1.0))
            ALGEBRAIST.compile_examples(probe, probe)
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
            ACCELERATOR.compile_examples(
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
            ACCELERATOR.compile_examples(
                _explicit_rhs_kernel,
                (
                    probe,
                    probe,
                    probe,
                    probe,
                    parameters.epsilon,
                    parameters.a,
                    parameters.b,
                ),
            )
            ACCELERATOR.compile_examples(
                _implicit_rhs_kernel,
                (
                    probe,
                    probe,
                    probe,
                    parameters.diffusivity_u,
                ),
            )
            self.__class__._compiled = True

    def __repr__(self) -> str:
        return "FitzHughNagumoDerivative()"

    __str__ = __repr__

    def __call__(self, interval: Interval, state: FitzHughNagumoState, out: FitzHughNagumoTranslation) -> None:
        del interval
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


class FitzHughNagumoExplicitDerivative:
    __slots__ = ("parameters",)

    def __init__(self, parameters: FitzHughNagumoParameters) -> None:
        self.parameters = parameters

    def __repr__(self) -> str:
        return "FitzHughNagumoExplicitDerivative()"

    __str__ = __repr__

    def __call__(self, interval: Interval, state: FitzHughNagumoState, out: FitzHughNagumoTranslation) -> None:
        del interval
        parameters = self.parameters
        _explicit_rhs_kernel(
            state.u,
            state.v,
            out.du,
            out.dv,
            parameters.epsilon,
            parameters.a,
            parameters.b,
        )


class FitzHughNagumoImplicitDerivative:
    __slots__ = ("parameters", "laplacian_u")

    def __init__(self, parameters: FitzHughNagumoParameters) -> None:
        self.parameters = parameters
        self.laplacian_u = np.zeros(parameters.grid_size, dtype=np.float64)

    def __repr__(self) -> str:
        return "FitzHughNagumoImplicitDerivative()"

    __str__ = __repr__

    def __call__(self, interval: Interval, state: FitzHughNagumoState, out: FitzHughNagumoTranslation) -> None:
        del interval
        parameters = self.parameters
        _laplacian_periodic(state.u, self.laplacian_u, parameters.inv_dx2)
        _implicit_rhs_kernel(
            self.laplacian_u,
            out.du,
            out.dv,
            parameters.diffusivity_u,
        )


class FitzHughNagumoLinearizer:
    __slots__ = ("parameters", "laplacian_du")

    def __init__(self, parameters: FitzHughNagumoParameters) -> None:
        self.parameters = parameters
        self.laplacian_du = np.zeros(parameters.grid_size, dtype=np.float64)

    def __repr__(self) -> str:
        return "FitzHughNagumoLinearizer()"

    __str__ = __repr__

    def __call__(self, interval, state, out):
        del interval
        parameters = self.parameters
        base_u = state.u

        def apply(translation, result):
            du = translation.du
            dv = translation.dv
            _laplacian_periodic(du, self.laplacian_du, parameters.inv_dx2)
            result.du[:] = parameters.diffusivity_u * self.laplacian_du + (1.0 - base_u * base_u) * du - dv
            result.dv[:] = parameters.epsilon * du - parameters.epsilon * parameters.b * dv

        out.apply = apply


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
    u_snapshots: list[np.ndarray]
    v_snapshots: list[np.ndarray]
    steps: int
    runtime: float


def _error_against_reference(state: FitzHughNagumoState, reference) -> float:
    du = state.u - reference["u"]
    dv = state.v - reference["v"]
    return float(np.sqrt((np.dot(du, du) + np.dot(dv, dv)) / (du.size + dv.size)))


def _scheme_regulator(scheme_type):
    if scheme_type is SchemeKennedyCarpenter43_7:
        return Regulator(safety=0.95, error_exponent=0.25)
    if scheme_type is SchemeKvaerno3:
        return Regulator(safety=0.95, error_exponent=1.0 / 3.0)
    return Regulator(safety=0.95, error_exponent=0.45)


def _imex_derivative(parameters: FitzHughNagumoParameters) -> ImExDerivative:
    return ImExDerivative(
        implicit=FitzHughNagumoImplicitDerivative(parameters),
        explicit=FitzHughNagumoExplicitDerivative(parameters),
    )


class FitzHughNagumoSpectralResolvent:
    __slots__ = ("tableau", "state", "operator_symbol", "u_hat")

    def __init__(self, parameters: FitzHughNagumoParameters, tableau=None) -> None:
        self.tableau = tableau
        self.state: FitzHughNagumoState | None = None
        theta = 2.0 * np.pi * np.fft.fftfreq(parameters.grid_size)
        self.operator_symbol = parameters.diffusivity_u * 2.0 * (np.cos(theta) - 1.0) * parameters.inv_dx2
        self.u_hat = np.zeros(parameters.grid_size, dtype=np.complex128)

    def __repr__(self) -> str:
        return f"FitzHughNagumoSpectralResolvent(tableau={self.tableau!r})"

    __str__ = __repr__

    def bind(self, interval: Interval, state: FitzHughNagumoState) -> None:
        del interval
        self.state = state

    def __call__(self, alpha: float, rhs, out) -> None:
        del rhs
        state = self.state
        if state is None:
            raise RuntimeError("FitzHughNagumoSpectralResolvent must be bound before use.")
        delta = out[0]
        if alpha == 0.0:
            delta.du.fill(0.0)
            delta.dv.fill(0.0)
            return

        np.copyto(self.u_hat, np.fft.fft(state.u))
        denominator = 1.0 - alpha * self.operator_symbol
        self.u_hat /= denominator
        resolved_u = np.fft.ifft(self.u_hat).real
        delta.du[:] = resolved_u - state.u
        delta.dv.fill(0.0)


def _build_anderson_solver(label, scheme_type, parameters: FitzHughNagumoParameters):
    safety = Safety.fast()
    workbench = FitzHughNagumoWorkbench(parameters.grid_size)
    derivative = FitzHughNagumoDerivative(parameters)
    resolvent = ResolventAnderson(
        derivative,
        workbench,
        _translation_inner_product,
        tolerance=ResolventTolerance(
            atol=parameters.resolution_atol,
            rtol=parameters.resolution_rtol,
        ),
        policy=ResolventPolicy(max_iterations=parameters.resolution_max_iterations),
        depth=4,
        safety=safety,
        accelerator=ACCELERATOR,
        tableau=scheme_type.tableau,
    )
    scheme = scheme_type(
        derivative,
        workbench,
        resolvent=resolvent,
        regulator=_scheme_regulator(scheme_type),
    )
    marcher = CountingMarcher(
        Marcher(
            scheme,
            Executor(
                tolerance=Tolerance(
                    atol=parameters.tolerance_atol,
                    rtol=parameters.tolerance_rtol,
                ),
                safety=safety,
                accelerator=ACCELERATOR,
            ),
        )
    )
    integrator = Integrator(executor=Executor(safety=safety, accelerator=ACCELERATOR))
    return label, integrator, marcher


def _build_imex_spectral_solver(label, scheme_type, parameters: FitzHughNagumoParameters):
    safety = Safety.fast()
    workbench = FitzHughNagumoWorkbench(parameters.grid_size)
    derivative = _imex_derivative(parameters)
    resolvent = FitzHughNagumoSpectralResolvent(parameters, tableau=scheme_type.tableau)
    scheme = scheme_type(
        derivative,
        workbench,
        resolvent=resolvent,
        regulator=_scheme_regulator(scheme_type),
    )
    marcher = CountingMarcher(
        Marcher(
            scheme,
            Executor(
                tolerance=Tolerance(
                    atol=parameters.tolerance_atol,
                    rtol=parameters.tolerance_rtol,
                ),
                safety=safety,
                accelerator=ACCELERATOR,
            ),
        )
    )
    integrator = Integrator(executor=Executor(safety=safety, accelerator=ACCELERATOR))
    return label, integrator, marcher


def prepare_quasi_newton(name, scheme_type, builder, problem_parameters, stark_parameters, initial_conditions, reference):
    parameters = parameters_from_benchmark(problem_parameters, stark_parameters)
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


def prepare_kvaerno3_anderson(problem_parameters, stark_parameters, initial_conditions, reference):
    return prepare_quasi_newton("Kvaerno3 Anderson", SchemeKvaerno3, _build_anderson_solver, problem_parameters, stark_parameters, initial_conditions, reference)


def prepare_kc43_imex_spectral(problem_parameters, stark_parameters, initial_conditions, reference):
    return prepare_quasi_newton(
        "KC43_7 IMEX Spectral",
        SchemeKennedyCarpenter43_7,
        _build_imex_spectral_solver,
        problem_parameters,
        stark_parameters,
        initial_conditions,
        reference,
    )


def run_inverter_example(name, inverter_class, parameters: FitzHughNagumoParameters) -> FitzHughNagumoTrajectory:
    del name
    safety = Safety.fast()
    workbench = FitzHughNagumoWorkbench(parameters.grid_size)
    derivative = FitzHughNagumoDerivative(parameters)
    linearizer = FitzHughNagumoLinearizer(parameters)
    inverter = inverter_class(
        workbench,
        _translation_inner_product,
        tolerance=InverterTolerance(
            atol=parameters.inversion_atol,
            rtol=parameters.inversion_rtol,
        ),
        policy=InverterPolicy(
            max_iterations=parameters.inversion_max_iterations,
            restart=parameters.inversion_restart,
        ),
        safety=safety,
        accelerator=ACCELERATOR,
    )
    resolvent = ResolventNewton(
        derivative,
        workbench,
        linearizer=linearizer,
        inverter=inverter,
        tolerance=ResolventTolerance(
            atol=parameters.resolution_atol,
            rtol=parameters.resolution_rtol,
        ),
        policy=ResolventPolicy(max_iterations=parameters.resolution_max_iterations),
        safety=safety,
        accelerator=ACCELERATOR,
        tableau=SchemeKvaerno3.tableau,
    )
    scheme = SchemeKvaerno3(
        derivative,
        workbench,
        resolvent=resolvent,
        regulator=_scheme_regulator(SchemeKvaerno3),
    )
    executor = Executor(
        tolerance=Tolerance(
            atol=parameters.tolerance_atol,
            rtol=parameters.tolerance_rtol,
        ),
        safety=safety,
        accelerator=ACCELERATOR,
    )
    integrate = Integrator(executor=executor)
    marcher = CountingMarcher(Marcher(scheme, executor))
    _x, state = initial_state(parameters)
    interval = Interval(parameters.t_start, parameters.initial_step, parameters.t_stop)

    u_snapshots = [state.u.copy()]
    v_snapshots = [state.v.copy()]
    started = perf_counter()
    for _interval, snapshot_state in integrate(
        marcher,
        interval,
        state,
        checkpoints=parameters.checkpoint_count,
    ):
        u_snapshots.append(snapshot_state.u.copy())
        v_snapshots.append(snapshot_state.v.copy())
    runtime = perf_counter() - started

    return FitzHughNagumoTrajectory(
        u_snapshots=u_snapshots,
        v_snapshots=v_snapshots,
        steps=marcher.steps,
        runtime=runtime,
    )











