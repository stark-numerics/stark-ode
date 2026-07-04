from __future__ import annotations

import pytest

from stark import Configuration, Tolerance
from stark.core.contracts import IntervalLike
from stark.core import IntegratorStepper
from stark.core.block import BlockBasis
from stark.core.interval import Interval
from stark.methods.inverters.dense import InverterDense
from stark.methods.inverters.relaxation import InverterRelaxationJacobi, InverterRelaxationRichardson
from stark.methods.resolvents import ResolventNewton
from stark.methods.schemes.implicit.fixed import SchemeBackwardEuler
from tests.support import (
    DummyVectorAllocator,
    DummyVectorBasis,
    DummyVectorState,
    DummyVectorTranslation,
)


class VectorDerivative:
    matrix = (
        (-10.0, 1.0, 0.0),
        (0.0, -8.0, 2.0),
        (0.0, 0.0, -5.0),
    )

    def __call__(
        self,
        interval: IntervalLike,
        state: DummyVectorState,
        out: DummyVectorTranslation,
    ) -> None:
        del interval
        out.values[:] = [
            sum(coefficient * state_value for coefficient, state_value in zip(row, state.values, strict=True))
            for row in self.matrix
        ]


class VectorLinearizer:
    matrix = VectorDerivative.matrix

    def __call__(self, interval: IntervalLike, state: DummyVectorState, out) -> None:
        del interval, state

        def apply(
            translation: DummyVectorTranslation,
            output: DummyVectorTranslation,
        ) -> None:
            output.values[:] = [
                sum(
                    coefficient * translation_value
                    for coefficient, translation_value in zip(row, translation.values, strict=True)
                )
                for row in self.matrix
            ]

        out.apply = apply


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


def materialize_entry(operator, basis: DummyVectorBasis) -> list[list[float]]:
    source = DummyVectorTranslation(*([0.0] * basis.dimension))
    image = DummyVectorTranslation(*([0.0] * basis.dimension))
    matrix = [[0.0 for _ in range(basis.dimension)] for _ in range(basis.dimension)]
    for column in range(basis.dimension):
        source = basis.vector(column, source)
        operator(source, image)
        for row in range(basis.dimension):
            matrix[row][column] = basis.coordinate(row, image)
    return matrix


def jacobi_entry_inverse(
    operator,
    defect: DummyVectorTranslation,
    output: DummyVectorTranslation,
) -> None:
    matrix = materialize_entry(operator, DummyVectorBasis(3))
    output.values[:] = solve_dense(matrix, defect.values)


def make_dense_inverter() -> InverterDense[DummyVectorTranslation]:
    return InverterDense(
        basis=BlockBasis([DummyVectorBasis(3)]),
    )


def make_jacobi_inverter() -> InverterRelaxationJacobi[DummyVectorTranslation]:
    return InverterRelaxationJacobi(
        diagonal_inverse=jacobi_entry_inverse,
        configuration=Configuration(inverter_tolerance=Tolerance(atol=1.0e-13, rtol=0.0), inverter_maximum_steps=4),
    )


def make_richardson_inverter() -> InverterRelaxationRichardson[DummyVectorTranslation]:
    # Backward Euler with step=0.1 builds approximately I - 0.1 J.  This
    # damping is conservative enough for the test matrix and deliberately keeps
    # Richardson as a genuine iterative cross-check rather than an exact solve.
    return InverterRelaxationRichardson(
        damping=0.5,
        configuration=Configuration(inverter_tolerance=Tolerance(atol=1.0e-11, rtol=0.0), inverter_maximum_steps=80),
    )


def run_backward_euler_newton(inverter) -> DummyVectorState:
    allocator = DummyVectorAllocator(3)
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
    state = DummyVectorState(1.0, -0.5, 0.25)

    stepper(interval, state)

    return state


def assert_state_close(actual: DummyVectorState, expected: DummyVectorState) -> None:
    for actual_value, expected_value in zip(actual.values, expected.values, strict=True):
        assert actual_value == pytest.approx(expected_value, abs=2.0e-9, rel=0.0)


def test_inverters_track_each_other_through_newton_resolvent_and_scheme() -> None:
    references: dict[str, DummyVectorState] = {}

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
