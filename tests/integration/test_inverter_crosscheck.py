from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import pytest

from stark import Tolerance
from stark.core import IntegratorStepper
from stark.core.block import Block, BlockBasis

try:  # New name after the diagonal block-operator rename.
    from stark.core.block import BlockOperatorDiagonal
except ImportError:  # Compatibility while developing this slow integration test.
    from stark.core.block import BlockOperator as BlockOperatorDiagonal

from stark.core.interval import Interval
from stark.methods.inverters.dense import InverterDense
from stark.methods.inverters.relaxation import InverterRelaxationJacobi, InverterRelaxationRichardson
from stark import Configuration, Tolerance
from stark.methods.resolvents import ResolventNewton
from stark import Configuration
from stark import Tolerance
from stark.methods.schemes.implicit.fixed import SchemeBackwardEuler


@dataclass(slots=True)
class VectorState:
    values: list[float]

    def __init__(self, values: list[float] | tuple[float, ...] | None = None) -> None:
        self.values = [0.0, 0.0, 0.0] if values is None else [float(value) for value in values]


@dataclass(slots=True)
class VectorTranslation:
    values: list[float]

    def __init__(self, values: list[float] | tuple[float, ...] | None = None) -> None:
        self.values = [0.0, 0.0, 0.0] if values is None else [float(value) for value in values]

    def __call__(self, origin: VectorState, result: VectorState) -> None:
        result.values[:] = [
            origin_value + shift_value
            for origin_value, shift_value in zip(origin.values, self.values, strict=True)
        ]

    def norm(self) -> float:
        return sqrt(sum(value * value for value in self.values))

    def __add__(self, other: VectorTranslation) -> VectorTranslation:
        return VectorTranslation([
            left + right
            for left, right in zip(self.values, other.values, strict=True)
        ])

    def __rmul__(self, scalar: float) -> VectorTranslation:
        return VectorTranslation([scalar * value for value in self.values])

    @staticmethod
    def scale(
        scalar: float,
        translation: VectorTranslation,
        output: VectorTranslation,
    ) -> VectorTranslation:
        output.values[:] = [scalar * value for value in translation.values]
        return output

    @staticmethod
    def combine2(
        left_scalar: float,
        left: VectorTranslation,
        right_scalar: float,
        right: VectorTranslation,
        output: VectorTranslation,
    ) -> VectorTranslation:
        output.values[:] = [
            left_scalar * left_value + right_scalar * right_value
            for left_value, right_value in zip(left.values, right.values, strict=True)
        ]
        return output

    @staticmethod
    def combine3(
        first_scalar: float,
        first: VectorTranslation,
        second_scalar: float,
        second: VectorTranslation,
        third_scalar: float,
        third: VectorTranslation,
        output: VectorTranslation,
    ) -> VectorTranslation:
        output.values[:] = [
            first_scalar * first_value
            + second_scalar * second_value
            + third_scalar * third_value
            for first_value, second_value, third_value in zip(
                first.values,
                second.values,
                third.values,
                strict=True,
            )
        ]
        return output

    linear_combine = [scale, combine2, combine3]


class VectorBasis:
    dimension = 3

    def vector(self, index: int, output: VectorTranslation) -> VectorTranslation:
        output.values[:] = [0.0, 0.0, 0.0]
        output.values[index] = 1.0
        return output

    def coordinate(self, index: int, translation: VectorTranslation) -> float:
        return translation.values[index]

    def coordinates(self, translation: VectorTranslation, output: list[float]) -> list[float]:
        output[:] = translation.values[:]
        return output

    def synthesize(
        self,
        coordinates: list[float],
        output: VectorTranslation,
    ) -> VectorTranslation:
        output.values[:] = list(coordinates)
        return output


class VectorAllocator:
    basis = VectorBasis()

    def allocate_state(self) -> VectorState:
        return VectorState()

    def copy_state(self, source: VectorState, output: VectorState) -> None:
        output.values[:] = source.values[:]

    def allocate_translation(self) -> VectorTranslation:
        return VectorTranslation()


class VectorDerivative:
    matrix = (
        (-10.0, 1.0, 0.0),
        (0.0, -8.0, 2.0),
        (0.0, 0.0, -5.0),
    )

    def __call__(self, interval: Interval, state: VectorState, output: VectorTranslation) -> None:
        del interval
        output.values[:] = [
            sum(coefficient * state_value for coefficient, state_value in zip(row, state.values, strict=True))
            for row in self.matrix
        ]


class VectorLinearizer:
    matrix = VectorDerivative.matrix

    def __call__(self, interval: Interval, state: VectorState, operator) -> None:
        del interval, state

        def apply(translation: VectorTranslation, output: VectorTranslation) -> None:
            output.values[:] = [
                sum(
                    coefficient * translation_value
                    for coefficient, translation_value in zip(row, translation.values, strict=True)
                )
                for row in self.matrix
            ]

        operator.apply = apply


def solve_dense(matrix: list[list[float]], residual: list[float]) -> list[float]:
    size = len(residual)
    augmented = [row[:] + [residual[index]] for index, row in enumerate(matrix)]

    for pivot in range(size):
        pivot_row = max(range(pivot, size), key=lambda row: abs(augmented[row][pivot]))
        pivot_value = augmented[pivot_row][pivot]
        if pivot_value == 0.0:
            raise ZeroDivisionError("singular dense test system")
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


def materialize_entry(operator, basis: VectorBasis) -> list[list[float]]:
    source = VectorTranslation()
    image = VectorTranslation()
    matrix = [[0.0 for _ in range(basis.dimension)] for _ in range(basis.dimension)]
    for column in range(basis.dimension):
        source = basis.vector(column, source)
        operator(source, image)
        for row in range(basis.dimension):
            matrix[row][column] = basis.coordinate(row, image)
    return matrix


def jacobi_entry_inverse(operator, defect: VectorTranslation, output: VectorTranslation) -> None:
    matrix = materialize_entry(operator, VectorBasis())
    output.values[:] = solve_dense(matrix, defect.values)


def make_dense_inverter() -> InverterDense[VectorTranslation]:
    return InverterDense(
        basis=BlockBasis([VectorBasis()]),
    )


def make_jacobi_inverter() -> InverterRelaxationJacobi[VectorTranslation]:
    return InverterRelaxationJacobi(
        diagonal_inverse=jacobi_entry_inverse,
        configuration=Configuration(inverter_tolerance=Tolerance(atol=1.0e-13, rtol=0.0), inverter_maximum_steps=4),
    )


def make_richardson_inverter() -> InverterRelaxationRichardson[VectorTranslation]:
    # Backward Euler with step=0.1 builds approximately I - 0.1 J.  This
    # damping is conservative enough for the test matrix and deliberately keeps
    # Richardson as a genuine iterative cross-check rather than an exact solve.
    return InverterRelaxationRichardson(
        damping=0.5,
        configuration=Configuration(inverter_tolerance=Tolerance(atol=1.0e-11, rtol=0.0), inverter_maximum_steps=80),
    )


def run_backward_euler_newton(inverter) -> VectorState:
    allocator = VectorAllocator()
    derivative = VectorDerivative()
    resolvent = ResolventNewton(
        allocator,
        linearizer=VectorLinearizer(),
        inverter=inverter,
        configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-10, rtol=1.0e-10), resolvent_maximum_steps=8),
    )
    scheme = SchemeBackwardEuler(
        derivative,
        allocator,
        resolvent=resolvent,
    )
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = VectorState((1.0, -0.5, 0.25))

    stepper(interval, state)

    return state


def assert_state_close(actual: VectorState, expected: VectorState) -> None:
    for actual_value, expected_value in zip(actual.values, expected.values, strict=True):
        assert actual_value == pytest.approx(expected_value, abs=2.0e-9, rel=0.0)


@pytest.mark.slow
def test_slow_new_inverters_track_each_other_through_newton_resolvent_and_scheme() -> None:
    references: dict[str, VectorState] = {}

    references["Dense"] = run_backward_euler_newton(make_dense_inverter())

    dense_reference = references["Dense"]
    candidates = {
        **references,
        "Jacobi": run_backward_euler_newton(make_jacobi_inverter()),
        "Richardson": run_backward_euler_newton(make_richardson_inverter()),
    }

    for label, state in candidates.items():
        del label
        assert_state_close(state, dense_reference)
