from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import pytest

from stark import Integrator, IntegratorStepper, Tolerance
from stark.core.interval import Interval
from stark.resolvents import (
    ResolventCoupledNewton,
    ResolventCoupledPicard,
    ResolventNewton,
    ResolventPicard,
)
from stark import Configuration
from stark import Tolerance
from stark.block import Block
from stark.schemes import SchemeBDF2, SchemeKvaerno3, SchemeKvaerno4, SchemeSDIRK21
from stark.schemes.implicit.fixed import (
    SchemeBackwardEuler,
    SchemeCrankNicolson,
    SchemeCrouzeixDIRK3,
    SchemeGaussLegendre4,
    SchemeImplicitMidpoint,
    SchemeLobattoIIIC4,
    SchemeRadauIIA5,
)


@dataclass(slots=True)
class ScalarTranslation:
    value: float = 0.0

    def __call__(self, origin: "ScalarState", result: "ScalarState") -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: "ScalarTranslation") -> "ScalarTranslation":
        return ScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> "ScalarTranslation":
        return ScalarTranslation(scalar * self.value)

    @staticmethod
    def scale(a: float, x: "ScalarTranslation", out: "ScalarTranslation") -> "ScalarTranslation":
        out.value = a * x.value
        return out

    @staticmethod
    def combine2(
        a0: float,
        x0: "ScalarTranslation",
        a1: float,
        x1: "ScalarTranslation",
        out: "ScalarTranslation",
    ) -> "ScalarTranslation":
        out.value = a0 * x0.value + a1 * x1.value
        return out

    linear_combine = [scale, combine2]


@dataclass(slots=True)
class ScalarState:
    value: float = 0.0


class ScalarAllocator:
    def allocate_state(self) -> ScalarState:
        return ScalarState()

    def copy_state(self, source: ScalarState, out: ScalarState) -> None:
        out.value = source.value

    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


class ScalarDerivative:
    def __init__(self, rate: float) -> None:
        self.rate = rate

    def __call__(self, interval: Interval, state: ScalarState, out: ScalarTranslation) -> None:
        del interval
        out.value = self.rate * state.value


class ScalarLinearizer:
    def __init__(self, rate: float) -> None:
        self.rate = rate

    def __call__(self, interval: Interval, state: ScalarState, out) -> None:
        del interval
        del state

        def apply(translation: ScalarTranslation, result: ScalarTranslation) -> None:
            result.value = self.rate * translation.value

        out.apply = apply


def scalar_inner_product(left: ScalarTranslation, right: ScalarTranslation) -> float:
    return left.value * right.value


class DenseScalarTestInverter:
    """Tiny exact scalar block inverter used to exercise the new request API."""

    def __call__(self, request, output: Block[ScalarTranslation]) -> None:
        size = len(request.residual)
        if len(output) != size:
            raise ValueError("DenseScalarTestInverter output size mismatch.")

        matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        for column in range(size):
            source = Block([ScalarTranslation() for _ in range(size)])
            image = Block([ScalarTranslation() for _ in range(size)])
            source[column].value = 1.0
            request.operator(source, image)
            for row in range(size):
                matrix[row][column] = image[row].value

        rhs = [request.residual[row].value for row in range(size)]
        solution = self.solve(matrix, rhs)
        for index, value in enumerate(solution):
            output[index].value = value

    @staticmethod
    def solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
        size = len(rhs)
        augmented = [row[:] + [rhs[index]] for index, row in enumerate(matrix)]

        for pivot in range(size):
            pivot_row = max(range(pivot, size), key=lambda row: abs(augmented[row][pivot]))
            if abs(augmented[pivot_row][pivot]) == 0.0:
                raise RuntimeError("DenseScalarTestInverter encountered a singular matrix.")
            if pivot_row != pivot:
                augmented[pivot], augmented[pivot_row] = augmented[pivot_row], augmented[pivot]

            pivot_value = augmented[pivot][pivot]
            for column in range(pivot, size + 1):
                augmented[pivot][column] /= pivot_value

            for row in range(size):
                if row == pivot:
                    continue
                factor = augmented[row][pivot]
                for column in range(pivot, size + 1):
                    augmented[row][column] -= factor * augmented[pivot][column]

        return [augmented[row][size] for row in range(size)]


def test_resolvent_tolerance_matches_general_tolerance_contract() -> None:
    tolerance = Tolerance(atol=1.0e-6, rtol=1.0e-3)

    assert tolerance.bound(2.0) == 0.002001
    assert tolerance.ratio(0.001, 2.0) < 1.0
    assert tolerance.accepts(0.001, 2.0)


def test_resolvent_picard_solves_scalar_backward_euler_step() -> None:
    allocator = ScalarAllocator()
    derivative = ScalarDerivative(rate=-1.0)
    resolvent = ResolventPicard(
        allocator,
        configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=32),
    )
    scheme = SchemeBackwardEuler(derivative, allocator, resolvent=resolvent)
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    stepper(interval, state)

    assert abs(state.value - (1.0 / 1.1)) < 1.0e-8
    assert interval.present == 0.1
    assert interval.step == 0.0


def test_backward_euler_matches_closed_form_for_quadratic_decay() -> None:
    class QuadraticDerivative:
        def __call__(self, interval: Interval, state: ScalarState, out: ScalarTranslation) -> None:
            del interval
            out.value = -(state.value ** 2)

    allocator = ScalarAllocator()
    scheme = SchemeBackwardEuler(
        QuadraticDerivative(),
        allocator,
        resolvent=ResolventPicard(
            allocator,
            configuration=Configuration(
                resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
                resolvent_maximum_steps=64,
            ),
        ),
    )
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    stepper(interval, state)

    expected = (-1.0 + sqrt(1.4)) / 0.2
    assert abs(state.value - expected) < 1.0e-10


def test_resolvent_newton_solves_scalar_backward_euler_step() -> None:
    allocator = ScalarAllocator()
    derivative = ScalarDerivative(rate=-10.0)
    inverter = DenseScalarTestInverter()
    resolvent = ResolventNewton(
        allocator,
        linearizer=ScalarLinearizer(rate=-10.0),
        inverter=inverter,
        configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=8),
    )
    scheme = SchemeBackwardEuler(
        derivative,
        allocator,
        resolvent=resolvent,
    )
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    stepper(interval, state)

    assert abs(state.value - 0.5) < 1.0e-10


def test_newton_resolvent_uses_explicitly_supplied_linearizer() -> None:
    allocator = ScalarAllocator()
    derivative = ScalarDerivative(rate=-10.0)
    inverter = DenseScalarTestInverter()
    resolvent = ResolventNewton(
        allocator,
        linearizer=ScalarLinearizer(rate=-10.0),
        inverter=inverter,
        configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=8),
    )
    scheme = SchemeBackwardEuler(derivative, allocator, resolvent=resolvent)
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    stepper(interval, state)

    assert abs(state.value - 0.5) < 1.0e-10


def test_resolvent_newton_requires_linearized_residual() -> None:
    allocator = ScalarAllocator()
    derivative = ScalarDerivative(rate=-1.0)
    inverter = DenseScalarTestInverter()
    with pytest.raises(TypeError):
        ResolventNewton(allocator, inverter=inverter)  # type: ignore[call-arg]

    resolvent = ResolventPicard(allocator)
    scheme = SchemeBackwardEuler(
        derivative,
        allocator,
        resolvent=resolvent,
    )
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    stepper(interval, state)
    assert abs(state.value - (1.0 / 1.1)) < 1.0e-8


def test_implicit_midpoint_solves_scalar_linear_decay_step() -> None:
    allocator = ScalarAllocator()
    derivative = ScalarDerivative(rate=-10.0)
    scheme = SchemeImplicitMidpoint(
        derivative,
        allocator,
        resolvent=ResolventPicard(
            allocator,
            configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=64),
            tableau=SchemeImplicitMidpoint.tableau,
        ),
    )
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    stepper(interval, state)

    assert abs(state.value - (1.0 / 3.0)) < 1.0e-10


def test_crank_nicolson_solves_scalar_linear_decay_step() -> None:
    allocator = ScalarAllocator()
    derivative = ScalarDerivative(rate=-10.0)
    scheme = SchemeCrankNicolson(
        derivative,
        allocator,
        resolvent=ResolventPicard(
            allocator,
            configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=64),
            tableau=SchemeCrankNicolson.tableau,
        ),
    )
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    stepper(interval, state)

    assert abs(state.value - (1.0 / 3.0)) < 1.0e-10


def test_crouzeix_dirk3_solves_constant_derivative_step() -> None:
    class ConstantDerivative:
        def __call__(self, interval: Interval, state: ScalarState, out: ScalarTranslation) -> None:
            del interval, state
            out.value = 1.0

    allocator = ScalarAllocator()
    derivative = ConstantDerivative()
    scheme = SchemeCrouzeixDIRK3(
        derivative,
        allocator,
        resolvent=ResolventPicard(
            allocator,
            configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=32),
            tableau=SchemeCrouzeixDIRK3.tableau,
        ),
    )
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    stepper(interval, state)

    assert abs(state.value - 1.1) < 1.0e-10


def test_gauss_legendre4_solves_constant_derivative_step() -> None:
    class ConstantDerivative:
        def __call__(self, interval: Interval, state: ScalarState, out: ScalarTranslation) -> None:
            del interval, state
            out.value = 1.0

    allocator = ScalarAllocator()
    derivative = ConstantDerivative()
    scheme = SchemeGaussLegendre4(
        derivative,
        allocator,
        resolvent=ResolventCoupledPicard(
            allocator,
            tableau=SchemeGaussLegendre4.tableau,
            configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=32),
        ),
    )
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    stepper(interval, state)

    assert abs(state.value - 1.1) < 1.0e-10


def test_coupled_newton_gauss_legendre4_solves_scalar_linear_decay_step() -> None:
    allocator = ScalarAllocator()
    derivative = ScalarDerivative(rate=-10.0)
    inverter = DenseScalarTestInverter()
    scheme = SchemeGaussLegendre4(
        derivative,
        allocator,
        resolvent=ResolventCoupledNewton(
            allocator,
            tableau=SchemeGaussLegendre4.tableau,
            linearizer=ScalarLinearizer(rate=-10.0),
            inverter=inverter,
            configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=8),
        ),
    )
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    stepper(interval, state)

    assert abs(state.value - (7.0 / 19.0)) < 1.0e-10


def test_radau_iia5_solves_constant_derivative_step() -> None:
    class ConstantDerivative:
        def __call__(self, interval: Interval, state: ScalarState, out: ScalarTranslation) -> None:
            del interval, state
            out.value = 1.0

    allocator = ScalarAllocator()
    derivative = ConstantDerivative()
    scheme = SchemeRadauIIA5(
        derivative,
        allocator,
        resolvent=ResolventCoupledPicard(
            allocator,
            tableau=SchemeRadauIIA5.tableau,
            configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=32),
        ),
    )
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    stepper(interval, state)

    assert abs(state.value - 1.1) < 1.0e-10


def test_lobatto_iiic4_solves_constant_derivative_step() -> None:
    class ConstantDerivative:
        def __call__(self, interval: Interval, state: ScalarState, out: ScalarTranslation) -> None:
            del interval, state
            out.value = 1.0

    allocator = ScalarAllocator()
    derivative = ConstantDerivative()
    scheme = SchemeLobattoIIIC4(
        derivative,
        allocator,
        resolvent=ResolventCoupledPicard(
            allocator,
            tableau=SchemeLobattoIIIC4.tableau,
            configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=32),
        ),
    )
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    stepper(interval, state)

    assert abs(state.value - 1.1) < 1.0e-10


def test_coupled_newton_radau_iia5_tracks_scalar_linear_decay() -> None:
    allocator = ScalarAllocator()
    derivative = ScalarDerivative(rate=-10.0)
    inverter = DenseScalarTestInverter()
    scheme = SchemeRadauIIA5(
        derivative,
        allocator,
        resolvent=ResolventCoupledNewton(
            allocator,
            tableau=SchemeRadauIIA5.tableau,
            linearizer=ScalarLinearizer(rate=-10.0),
            inverter=inverter,
            configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=8),
        ),
    )
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    stepper(interval, state)

    assert abs(state.value - 0.36787944117144233) < 1.0e-3


def test_coupled_newton_lobatto_iiic4_tracks_scalar_linear_decay() -> None:
    allocator = ScalarAllocator()
    derivative = ScalarDerivative(rate=-10.0)
    inverter = DenseScalarTestInverter()
    scheme = SchemeLobattoIIIC4(
        derivative,
        allocator,
        resolvent=ResolventCoupledNewton(
            allocator,
            tableau=SchemeLobattoIIIC4.tableau,
            linearizer=ScalarLinearizer(rate=-10.0),
            inverter=inverter,
            configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=8),
        ),
    )
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    stepper(interval, state)

    assert abs(state.value - 0.36787944117144233) < 1.0e-3


def test_backward_euler_rejects_mismatched_resolvent_tableau() -> None:
    allocator = ScalarAllocator()
    derivative = ScalarDerivative(rate=-1.0)

    with pytest.raises(ValueError):
        SchemeBackwardEuler(
            derivative,
            allocator,
            resolvent=ResolventPicard(
                allocator,
                tableau=SchemeSDIRK21.tableau,
            ),
        )


def test_sdirk21_advances_linear_decay_with_adaptive_control() -> None:
    allocator = ScalarAllocator()
    derivative = ScalarDerivative(rate=-10.0)
    inverter = DenseScalarTestInverter()
    resolvent = ResolventNewton(
        allocator,
        linearizer=ScalarLinearizer(rate=-10.0),
        inverter=inverter,
        configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=8),
    )
    scheme = SchemeSDIRK21(
        derivative,
        allocator,
        resolvent=resolvent,
    )
    stepper = IntegratorStepper(scheme)
    integrate = Integrator()
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    for _interval, _state in integrate.live(stepper, interval, state):
        pass

    assert abs(state.value - 0.36787944117144233) < 5.0e-4


def test_kvaerno3_advances_linear_decay_with_adaptive_control() -> None:
    allocator = ScalarAllocator()
    derivative = ScalarDerivative(rate=-10.0)
    inverter = DenseScalarTestInverter()
    resolvent = ResolventNewton(
        allocator,
        linearizer=ScalarLinearizer(rate=-10.0),
        inverter=inverter,
        configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=8),
    )
    scheme = SchemeKvaerno3(
        derivative,
        allocator,
        resolvent=resolvent,
    )
    stepper = IntegratorStepper(scheme)
    integrate = Integrator()
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    for _interval, _state in integrate.live(stepper, interval, state):
        pass

    assert abs(state.value - 0.36787944117144233) < 1.0e-4


def test_kvaerno4_advances_linear_decay_with_adaptive_control() -> None:
    allocator = ScalarAllocator()
    derivative = ScalarDerivative(rate=-10.0)
    inverter = DenseScalarTestInverter()
    resolvent = ResolventNewton(
        allocator,
        linearizer=ScalarLinearizer(rate=-10.0),
        inverter=inverter,
        configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=8),
    )
    scheme = SchemeKvaerno4(
        derivative,
        allocator,
        resolvent=resolvent,
    )
    stepper = IntegratorStepper(scheme)
    integrate = Integrator()
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    for _interval, _state in integrate.live(stepper, interval, state):
        pass

    assert abs(state.value - 0.36787944117144233) < 5.0e-5


def test_bdf2_advances_linear_decay_with_adaptive_control() -> None:
    allocator = ScalarAllocator()
    derivative = ScalarDerivative(rate=-10.0)
    inverter = DenseScalarTestInverter()
    resolvent = ResolventNewton(
        allocator,
        linearizer=ScalarLinearizer(rate=-10.0),
        inverter=inverter,
        configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=8),
    )
    scheme = SchemeBDF2(
        derivative,
        allocator,
        resolvent=resolvent,
    )
    stepper = IntegratorStepper(scheme)
    integrate = Integrator()
    interval = Interval(present=0.0, step=0.05, stop=0.2)
    state = ScalarState(1.0)

    for _interval, _state in integrate.live(stepper, interval, state):
        pass

    assert abs(state.value - 0.1353352832366127) < 2.0e-2
