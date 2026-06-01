from __future__ import annotations

from dataclasses import dataclass
from math import acos, copysign, cos, pi, sqrt
from typing import Any

import numpy as np

from stark import Executor, ExecutorSafety, ExecutorTolerance, Interval, Marcher
from stark.accelerators import Accelerator
from stark.algebraist.generator import AlgebraistGeneratorSpecialist
from stark.algebraist.layout import AlgebraistLayout, AlgebraistLayoutField, AlgebraistLayoutUnravel
from stark.block import BlockBasis, BlockSpecialist
from stark.interface import StarkDerivative, StarkIVP, StarkVector
from stark.interface.vector import StarkVectorAllocator, StarkVectorTranslation
from stark.inverters import InverterRelaxationJacobi
from stark.inverters.dense import InverterDense, InverterProviderDenseNative
from stark.inverters.support import InverterBudget, InverterTolerance
from stark.monitor import MonitorInverter
from stark.resolvents import ResolventNewton, ResolventPolicy, ResolventTolerance
from stark.schemes import SchemeKvaerno4

try:
    ACCELERATOR = Accelerator.numba()
    USE_NUMBA_ACCELERATION = True
except ModuleNotFoundError:
    ACCELERATOR = Accelerator.none()
    USE_NUMBA_ACCELERATION = False

Array = Any

ROBERTSON_LAYOUT = AlgebraistLayout(
    fields=(AlgebraistLayoutField("value.dy", "value.y", policy=AlgebraistLayoutUnravel(shape=(3,))),),
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
def _jacobian_kernel(state_y: Array, dy: Array, out_y: Array) -> None:
    y2 = state_y[1]
    y3 = state_y[2]
    coupling = 1.0e4 * (dy[1] * y3 + y2 * dy[2])
    quadratic = 6.0e7 * y2 * dy[1]
    out_y[0] = -0.04 * dy[0] + coupling
    out_y[1] = 0.04 * dy[0] - coupling - quadratic
    out_y[2] = quadratic


@ACCELERATOR.compile
def _state_error_kernel(y: Array, reference_y: Array) -> float:
    dy0 = y[0] - reference_y[0]
    dy1 = y[1] - reference_y[1]
    dy2 = y[2] - reference_y[2]
    return ((dy0 * dy0 + dy1 * dy1 + dy2 * dy2) / 3.0) ** 0.5


@dataclass(slots=True)
class RobertsonState:
    y: np.ndarray


@dataclass(slots=True)
class RobertsonTranslation:
    dy: np.ndarray

    def __call__(self, origin: RobertsonState, result: RobertsonState) -> None:
        np.add(origin.y, self.dy, out=result.y)

    def norm(self) -> float:
        return float(((self.dy[0] * self.dy[0] + self.dy[1] * self.dy[1] + self.dy[2] * self.dy[2]) / 3.0) ** 0.5)


class RobertsonCarrierValidation:
    def validate_state(self, value: RobertsonState) -> RobertsonState:
        if not isinstance(value, RobertsonState):
            raise TypeError("Robertson state must be a RobertsonState.")
        return value

    def validate_translation(self, value: RobertsonTranslation) -> RobertsonTranslation:
        if not isinstance(value, RobertsonTranslation):
            raise TypeError("Robertson translation must be a RobertsonTranslation.")
        return value

    def coerce_translation(self, value: object) -> RobertsonTranslation:
        return self.validate_translation(value)  # type: ignore[arg-type]


class RobertsonCarrierAllocation:
    def zero_state(self) -> RobertsonState:
        return RobertsonState(np.zeros(3, dtype=np.float64))

    def zero_translation(self) -> RobertsonTranslation:
        return RobertsonTranslation(np.zeros(3, dtype=np.float64))

    def allocate_translation(self) -> RobertsonTranslation:
        return self.zero_translation()

    def copy_state(self, value: RobertsonState) -> RobertsonState:
        return RobertsonState(value.y.copy())

    def copy_translation(self, value: RobertsonTranslation) -> RobertsonTranslation:
        return RobertsonTranslation(value.dy.copy())


class RobertsonCarrierArithmetic:
    preference = "into"

    def translate(
        self,
        state: RobertsonState,
        step: float,
        derivative: RobertsonTranslation,
        result: RobertsonState,
    ) -> None:
        result.y[:] = state.y + step * derivative.dy

    def add(self, left: RobertsonTranslation, right: RobertsonTranslation, result: RobertsonTranslation) -> None:
        np.add(left.dy, right.dy, out=result.dy)

    def scale(self, factor: float, value: RobertsonTranslation, result: RobertsonTranslation) -> None:
        np.multiply(value.dy, factor, out=result.dy)

    def add_scaled(self, result: RobertsonTranslation, factor: float, value: RobertsonTranslation) -> None:
        result.dy += factor * value.dy

    def combine2(self, a0, x0, a1, x1, result):
        np.multiply(x0.dy, a0, out=result.dy)
        self.add_scaled(result, a1, x1)

    def combine3(self, a0, x0, a1, x1, a2, x2, result):
        self.combine2(a0, x0, a1, x1, result)
        self.add_scaled(result, a2, x2)

    def combine4(self, a0, x0, a1, x1, a2, x2, a3, x3, result):
        self.combine3(a0, x0, a1, x1, a2, x2, result)
        self.add_scaled(result, a3, x3)

    def combine5(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, result):
        self.combine4(a0, x0, a1, x1, a2, x2, a3, x3, result)
        self.add_scaled(result, a4, x4)

    def combine6(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, result):
        self.combine5(a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, result)
        self.add_scaled(result, a5, x5)

    def combine7(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, result):
        self.combine6(a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, result)
        self.add_scaled(result, a6, x6)

    def combine8(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, result):
        self.combine7(a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, result)
        self.add_scaled(result, a7, x7)

    def combine9(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, result):
        self.combine8(a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, result)
        self.add_scaled(result, a8, x8)

    def combine10(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, result):
        self.combine9(a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, result)
        self.add_scaled(result, a9, x9)

    def combine11(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, a10, x10, result):
        self.combine10(a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, result)
        self.add_scaled(result, a10, x10)

    def combine12(self, a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, a10, x10, a11, x11, result):
        self.combine11(a0, x0, a1, x1, a2, x2, a3, x3, a4, x4, a5, x5, a6, x6, a7, x7, a8, x8, a9, x9, a10, x10, result)
        self.add_scaled(result, a11, x11)


class RobertsonCarrier:
    def __init__(self) -> None:
        self.validation = RobertsonCarrierValidation()
        self.allocation = RobertsonCarrierAllocation()
        self.arithmetic = RobertsonCarrierArithmetic()
        self.norm = lambda value: value.norm()


class RobertsonFullDerivative:
    __slots__ = ()
    _compiled = False

    def __init__(self) -> None:
        if not self.__class__._compiled:
            probe = np.zeros(3, dtype=np.float64)
            ACCELERATOR.compile_examples(_full_rhs_kernel, (probe, probe))
            self.__class__._compiled = True

    def __call__(self, _time: float, state: RobertsonState, out: RobertsonTranslation) -> None:
        _full_rhs_kernel(state.y, out.dy)


class RobertsonFullLinearizer:
    __slots__ = ()
    _compiled = False

    def __init__(self) -> None:
        if not self.__class__._compiled:
            probe = np.zeros(3, dtype=np.float64)
            ACCELERATOR.compile_examples(_jacobian_kernel, (probe, probe, probe))
            self.__class__._compiled = True

    def __call__(self, interval, state: StarkVector, out) -> None:
        del interval
        state_y = state.value.y

        def apply(translation: StarkVectorTranslation, result: StarkVectorTranslation) -> None:
            _jacobian_kernel(state_y, translation.value.dy, result.value.dy)

        out.apply = apply


class RobertsonVectorBasis:
    """Coordinate basis for the wrapped Robertson vector translation."""

    dimension = 3

    def vector(self, index: int, output: StarkVectorTranslation) -> StarkVectorTranslation:
        output.value.dy[:] = 0.0
        output.value.dy[index] = 1.0
        return output

    def coordinate(self, index: int, translation: StarkVectorTranslation) -> float:
        return float(translation.value.dy[index])

    def coordinates(self, translation: StarkVectorTranslation, output: list[float]) -> list[float]:
        output[:] = [float(value) for value in translation.value.dy]
        return output

    def synthesize(self, coordinates: list[float], output: StarkVectorTranslation) -> StarkVectorTranslation:
        output.value.dy[:] = coordinates
        return output


class RobertsonEntryOperatorInverse:
    __slots__ = ("basis", "image", "matrix")

    def __init__(self, allocator: StarkVectorAllocator) -> None:
        self.basis = allocator.allocate_translation()
        self.image = allocator.allocate_translation()
        self.matrix = np.zeros((3, 3), dtype=np.float64)

    def __call__(self, operator, source: StarkVectorTranslation, target: StarkVectorTranslation) -> None:
        basis = self.basis
        image = self.image
        matrix = self.matrix

        for column in range(3):
            basis.value.dy[:] = 0.0
            basis.value.dy[column] = 1.0
            operator(basis, image)
            matrix[:, column] = image.value.dy

        target.value.dy[:] = np.linalg.solve(matrix, source.value.dy)


class RobertsonFullCubicResolvent:
    __slots__ = ("tableau", "cubic")

    def __init__(self, tableau=None) -> None:
        self.tableau = tableau
        self.cubic = RobertsonCubicRoot()

    def bind(self, interval, state) -> None:
        del interval, state

    def __call__(self, problem, out) -> None:
        state_y = problem.origin.value.y
        alpha = problem.alpha
        rhs = problem.rhs

        delta = out[0]
        if alpha == 0.0:
            if rhs is None:
                delta.value.dy[:] = 0.0
            else:
                np.copyto(delta.value.dy, rhs[0].value.dy)
            return

        rhs_y = rhs[0].value.dy if rhs is not None else None
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

        delta.value.dy[0] = z1 - state_y[0]
        delta.value.dy[1] = z2 - state_y[1]
        delta.value.dy[2] = z3 - state_y[2]


class RobertsonCubicRoot:
    __slots__ = ()

    @staticmethod
    def _cbrt(value: float) -> float:
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
            root = self._cbrt(-half_q + sqrt(discriminant)) + self._cbrt(-half_q - sqrt(discriminant))
            return root - a / 3.0

        radius = 2.0 * sqrt(-third_p)
        angle = acos(max(-1.0, min(1.0, -half_q / sqrt(-(third_p * third_p * third_p)))))
        roots = [radius * cos((angle + 2.0 * pi * index) / 3.0) - a / 3.0 for index in range(3)]
        bounded = [root for root in roots if lower - 1.0e-12 <= root <= upper + 1.0e-12]
        if bounded:
            return min(bounded, key=lambda root: abs(root - target))
        return min(roots, key=lambda root: abs(root - target))


@dataclass(slots=True)
class RobertsonStarkSolver:
    name: str
    build: Any
    marcher: Marcher
    problem_parameters: dict[str, float]
    initial_conditions: dict[str, np.ndarray]
    reference: dict[str, Any]
    diagnostics: Any | None = None

    def __call__(self) -> dict[str, Any]:
        interval = self.build.interval.copy()
        state = StarkVector(
            RobertsonState(self.initial_conditions["y"].copy()),
            self.build.initial.carrier,
        )
        steps = 0

        for _interval, _state in self.build.integrator.live(self.marcher, interval, state):
            steps += 1

        result: dict[str, Any] = {
            "library": "STARK",
            "solver": self.name,
            "error": float(_state_error_kernel(state.value.y, self.reference["y"])),
            "steps": steps,
        }
        if self.diagnostics is not None:
            result.update(self.diagnostics())
        return result


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


def build_template(problem_parameters, stark_parameters, initial_conditions):
    return StarkIVP(
        derivative=StarkDerivative.in_place(RobertsonFullDerivative()),
        initial=RobertsonState(initial_conditions["y"].copy()),
        interval=Interval(problem_parameters["t0"], stark_parameters["step"], problem_parameters["t1"]),
        carrier=RobertsonCarrier(),
        executor=Executor(
            tolerance=ExecutorTolerance(
                atol=stark_parameters["tolerance_atol"],
                rtol=stark_parameters["tolerance_rtol"],
            ),
            safety=ExecutorSafety.fast(),
        ),
    ).build()


def build_specialist(allocator: StarkVectorAllocator):
    return AlgebraistGeneratorSpecialist(
        translation=allocator.allocate_translation(),
        allocator=allocator,
        layout=ROBERTSON_LAYOUT,
        accelerator=ACCELERATOR,
    )


def build_scheme(build, resolvent, specialist):
    return SchemeKvaerno4(build.derivative, build.allocator, resolvent=resolvent, specialist=specialist)


def build_cubic_resolvent():
    return RobertsonFullCubicResolvent(tableau=SchemeKvaerno4.tableau)


def build_newton_jacobi_inverter(allocator, stark_parameters, monitor: MonitorInverter, specialist):
    return InverterRelaxationJacobi(
        RobertsonEntryOperatorInverse(allocator),
        damping=1.0,
        tolerance=InverterTolerance(
            atol=stark_parameters["inversion_atol"],
            rtol=stark_parameters["inversion_rtol"],
        ),
        budget=InverterBudget(maximum_steps=stark_parameters["inversion_max_iterations"]),
        monitor=monitor,
        specialist=BlockSpecialist(specialist),
    )


def build_newton_jacobi_resolvent(build, stark_parameters, monitor: MonitorInverter, specialist):
    return ResolventNewton(
        build.allocator,
        linearizer=RobertsonFullLinearizer(),
        inverter=build_newton_jacobi_inverter(build.allocator, stark_parameters, monitor, specialist),
        ExecutorTolerance=ResolventTolerance(
            atol=stark_parameters["resolution_atol"],
            rtol=stark_parameters["resolution_rtol"],
        ),
        policy=ResolventPolicy(max_iterations=stark_parameters["resolution_max_iterations"]),
        safety=ExecutorSafety.fast(),
        accelerator=ACCELERATOR,
        specialist=BlockSpecialist(specialist),
        tableau=SchemeKvaerno4.tableau,
    )


def build_newton_dense_inverter(monitor: MonitorInverter):
    return InverterDense(
        basis=BlockBasis([RobertsonVectorBasis()]),
        provider=InverterProviderDenseNative(accelerator=ACCELERATOR),
        monitor=monitor,
    )


def build_newton_dense_resolvent(build, stark_parameters, monitor: MonitorInverter, specialist):
    return ResolventNewton(
        build.allocator,
        linearizer=RobertsonFullLinearizer(),
        inverter=build_newton_dense_inverter(monitor),
        ExecutorTolerance=ResolventTolerance(
            atol=stark_parameters["resolution_atol"],
            rtol=stark_parameters["resolution_rtol"],
        ),
        policy=ResolventPolicy(max_iterations=stark_parameters["resolution_max_iterations"]),
        safety=ExecutorSafety.fast(),
        accelerator=ACCELERATOR,
        specialist=BlockSpecialist(specialist),
        tableau=SchemeKvaerno4.tableau,
    )


def prepare_kvaerno4_full_custom(problem_parameters, stark_parameters, initial_conditions, reference):
    build = build_template(problem_parameters, stark_parameters, initial_conditions)
    specialist = build_specialist(build.allocator)
    scheme = build_scheme(build, build_cubic_resolvent(), specialist)
    return RobertsonStarkSolver(
        "Kvaerno4 Full Cubic",
        build,
        Marcher(scheme, build.executor),
        problem_parameters,
        initial_conditions,
        reference,
    )


def prepare_kvaerno4_full_newton(problem_parameters, stark_parameters, initial_conditions, reference):
    build = build_template(problem_parameters, stark_parameters, initial_conditions)
    specialist = build_specialist(build.allocator)
    inverter_monitor = MonitorInverter()
    resolvent = build_newton_jacobi_resolvent(build, stark_parameters, inverter_monitor, specialist)
    scheme = build_scheme(build, resolvent, specialist)
    return RobertsonStarkSolver(
        "Kvaerno4 Full Newton Jacobi",
        build,
        Marcher(scheme, build.executor),
        problem_parameters,
        initial_conditions,
        reference,
        diagnostics=RobertsonInverterDiagnostics(inverter_monitor),
    )


def prepare_kvaerno4_full_newton_dense(problem_parameters, stark_parameters, initial_conditions, reference):
    build = build_template(problem_parameters, stark_parameters, initial_conditions)
    specialist = build_specialist(build.allocator)
    inverter_monitor = MonitorInverter()
    resolvent = build_newton_dense_resolvent(build, stark_parameters, inverter_monitor, specialist)
    scheme = build_scheme(build, resolvent, specialist)
    return RobertsonStarkSolver(
        "Kvaerno4 Full Newton Dense",
        build,
        Marcher(scheme, build.executor),
        problem_parameters,
        initial_conditions,
        reference,
        diagnostics=RobertsonInverterDiagnostics(inverter_monitor),
    )
