from __future__ import annotations

from collections.abc import Sequence

import pytest

from stark.core.block import Block, BlockBasis
from stark.core.contracts import (
    BlockLike,
    BlockOperatorDiagonalLike,
    BlockOperatorEntryLike,
)
from stark.core.contracts.inverter import InverterRequest
from stark.methods.inverters.dense import InverterDense, InverterDenseInstanceSingle
from tests.support import (
    DummyVectorBasis,
    DummyVectorTranslation,
    assert_dummy_vector_close,
)


class DummyOperatorDiagonal:
    """Block-diagonal operator with entries supplied directly by the test."""

    def __init__(
        self,
        entries: Sequence[BlockOperatorEntryLike[DummyVectorTranslation] | None],
    ) -> None:
        self.entries = list(entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(
        self,
        index: int,
    ) -> BlockOperatorEntryLike[DummyVectorTranslation] | None:
        return self.entries[index]

    def __call__(
        self,
        source: BlockLike[DummyVectorTranslation],
        target: BlockLike[DummyVectorTranslation],
    ) -> BlockLike[DummyVectorTranslation]:
        for index, entry in enumerate(self.entries):
            if entry is None:
                raise RuntimeError("missing entry")
            entry(source[index], target[index])
        return target


class DummyInverterRequest:
    """Minimal request object for dense inverter calls."""

    def __init__(
        self,
        operator: BlockOperatorDiagonalLike[DummyVectorTranslation],
        residual: BlockLike[DummyVectorTranslation],
    ) -> None:
        self.operator = operator
        self.residual = residual


def matrix_operator(
    matrix: list[list[float]],
) -> BlockOperatorEntryLike[DummyVectorTranslation]:
    """Return a dense matrix action over `DummyVectorTranslation`."""

    def apply(
        source: DummyVectorTranslation,
        target: DummyVectorTranslation,
    ) -> None:
        target.values[:] = [
            sum(
                coefficient * source_value
                for coefficient, source_value in zip(row, source.values, strict=True)
            )
            for row in matrix
        ]

    return apply


def test_dense_inverter_solves_scalar_problem() -> None:
    basis = BlockBasis([DummyVectorBasis(1)])
    inverter = InverterDense(basis=basis)
    request = dense_request(
        operator=DummyOperatorDiagonal([matrix_operator([[4.0]])]),
        residual=Block([DummyVectorTranslation(8.0)]),
    )
    output = Block([DummyVectorTranslation(0.0)])

    inverter(request, output)

    assert_dummy_vector_close(output[0], [2.0])


def test_dense_inverter_solves_two_by_two_problem() -> None:
    basis = BlockBasis([DummyVectorBasis(2)])
    inverter = InverterDense(basis=basis)
    request = dense_request(
        operator=DummyOperatorDiagonal([matrix_operator([[2.0, 1.0], [1.0, 3.0]])]),
        residual=Block([DummyVectorTranslation(1.0, 2.0)]),
    )
    output = Block([DummyVectorTranslation(0.0, 0.0)])

    inverter(request, output)

    assert_dummy_vector_close(output[0], [0.2, 0.6])


def test_dense_inverter_solves_three_by_three_problem() -> None:
    basis = BlockBasis([DummyVectorBasis(3)])
    inverter = InverterDense(basis=basis)
    request = dense_request(
        operator=DummyOperatorDiagonal([
            matrix_operator([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]])
        ]),
        residual=Block([DummyVectorTranslation(14.0, 14.0, 17.0)]),
    )
    output = Block([DummyVectorTranslation(0.0, 0.0, 0.0)])

    inverter(request, output)

    assert_dummy_vector_close(output[0], [1.0, 2.0, 3.0])


def test_dense_inverter_solves_blockwise_larger_system() -> None:
    basis = BlockBasis([DummyVectorBasis(2), DummyVectorBasis(2)])
    inverter = InverterDense(basis=basis)
    request = dense_request(
        operator=DummyOperatorDiagonal([
            matrix_operator([[2.0, 0.0], [0.0, 4.0]]),
            matrix_operator([[5.0, 0.0], [0.0, 10.0]]),
        ]),
        residual=Block([DummyVectorTranslation(6.0, 8.0), DummyVectorTranslation(15.0, 40.0)]),
    )
    output = Block([DummyVectorTranslation(0.0, 0.0), DummyVectorTranslation(0.0, 0.0)])

    inverter(request, output)

    assert_dummy_vector_close(output[0], [3.0, 2.0])
    assert_dummy_vector_close(output[1], [3.0, 4.0])


def test_dense_inverter_rejects_singular_exact_system() -> None:
    basis = BlockBasis([DummyVectorBasis(2)])
    inverter = InverterDense(basis=basis)
    request = dense_request(
        operator=DummyOperatorDiagonal([matrix_operator([[1.0, 2.0], [2.0, 4.0]])]),
        residual=Block([DummyVectorTranslation(1.0, 1.0)]),
    )

    with pytest.raises(ZeroDivisionError):
        inverter(request, Block([DummyVectorTranslation(0.0, 0.0)]))


def dense_fill_operator(
    matrix: list[list[float]],
) -> BlockOperatorEntryLike[DummyVectorTranslation]:
    """Return a matrix action that also exposes dense materialisation."""

    def apply(
        source: DummyVectorTranslation,
        target: DummyVectorTranslation,
    ) -> None:
        target.values[:] = [
            sum(
                coefficient * source_value
                for coefficient, source_value in zip(row, source.values, strict=True)
            )
            for row in matrix
        ]

    def dense_fill(
        _basis,
        target: list[float],
        row_offset: int,
        column_offset: int,
        stride: int,
    ) -> None:
        for row, values in enumerate(matrix):
            for column, value in enumerate(values):
                target[(row_offset + row) * stride + column_offset + column] = value

    apply.dense_fill = dense_fill  # type: ignore[attr-defined]
    return apply


def dense_request(
    operator: BlockOperatorDiagonalLike[DummyVectorTranslation],
    residual: BlockLike[DummyVectorTranslation],
) -> InverterRequest[DummyVectorTranslation]:
    """Build a dense inverter request with its protocol shape made explicit."""

    return DummyInverterRequest(operator, residual)


def test_dense_inverter_instance_reuses_materialized_block_matrix() -> None:
    basis = BlockBasis([DummyVectorBasis(3)])
    inverter = InverterDense(basis=basis)
    operator = DummyOperatorDiagonal([
        dense_fill_operator([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]])
    ])
    instance = inverter.instance(operator)
    output = Block([DummyVectorTranslation(0.0, 0.0, 0.0)])

    assert isinstance(instance, InverterDenseInstanceSingle)
    instance(Block([DummyVectorTranslation(14.0, 14.0, 17.0)]), output)

    assert_dummy_vector_close(output[0], [1.0, 2.0, 3.0])


def test_dense_inverter_instance_requires_dense_fill_entries() -> None:
    basis = BlockBasis([DummyVectorBasis(2)])
    inverter = InverterDense(basis=basis)
    operator = DummyOperatorDiagonal([matrix_operator([[2.0, 1.0], [1.0, 3.0]])])

    with pytest.raises(TypeError, match="dense_fill"):
        inverter.instance(operator)


class DummyMonitor:
    """Capture dense inverter monitor records without depending on Monitor."""

    def __init__(self) -> None:
        self.solves = []

    def record_solve(
        self,
        inverter,
        converged,
        iteration_count,
        initial_residual,
        final_residual,
        failure_reason,
    ) -> None:
        self.solves.append(
            (inverter, converged, iteration_count, initial_residual, final_residual, failure_reason)
        )


def test_dense_inverter_unmonitored_path_does_not_call_record(monkeypatch) -> None:
    def fail_record(*_args, **_kwargs):
        raise AssertionError("unmonitored dense path should not call record_solve")

    monkeypatch.setattr(InverterDense, "record_solve", fail_record)
    basis = BlockBasis([DummyVectorBasis(2)])
    inverter = InverterDense(basis=basis)
    request = dense_request(
        operator=DummyOperatorDiagonal([matrix_operator([[2.0, 0.0], [0.0, 4.0]])]),
        residual=Block([DummyVectorTranslation(6.0, 8.0)]),
    )
    output = Block([DummyVectorTranslation(0.0, 0.0)])

    inverter(request, output)

    assert_dummy_vector_close(output[0], [3.0, 2.0])


def test_dense_inverter_monitored_path_records_once() -> None:
    monitor = DummyMonitor()
    basis = BlockBasis([DummyVectorBasis(2)])
    inverter = InverterDense(basis=basis, monitor=monitor)
    request = dense_request(
        operator=DummyOperatorDiagonal([matrix_operator([[2.0, 0.0], [0.0, 4.0]])]),
        residual=Block([DummyVectorTranslation(6.0, 8.0)]),
    )
    output = Block([DummyVectorTranslation(0.0, 0.0)])

    inverter(request, output)

    assert_dummy_vector_close(output[0], [3.0, 2.0])
    assert monitor.solves == [("Dense", True, None, None, None, None)]


def test_dense_inverter_instance_unmonitored_path_does_not_call_record(monkeypatch) -> None:
    def fail_record(*_args, **_kwargs):
        raise AssertionError("unmonitored dense instance path should not call record_solve")

    monkeypatch.setattr(InverterDense, "record_solve", fail_record)
    basis = BlockBasis([DummyVectorBasis(2)])
    inverter = InverterDense(basis=basis)
    operator = DummyOperatorDiagonal([dense_fill_operator([[2.0, 0.0], [0.0, 4.0]])])
    instance = inverter.instance(operator)
    output = Block([DummyVectorTranslation(0.0, 0.0)])

    instance(Block([DummyVectorTranslation(6.0, 8.0)]), output)

    assert_dummy_vector_close(output[0], [3.0, 2.0])


def test_dense_inverter_instance_monitored_path_records_once() -> None:
    monitor = DummyMonitor()
    basis = BlockBasis([DummyVectorBasis(2)])
    inverter = InverterDense(basis=basis, monitor=monitor)
    operator = DummyOperatorDiagonal([dense_fill_operator([[2.0, 0.0], [0.0, 4.0]])])
    instance = inverter.instance(operator)
    output = Block([DummyVectorTranslation(0.0, 0.0)])

    instance(Block([DummyVectorTranslation(6.0, 8.0)]), output)

    assert_dummy_vector_close(output[0], [3.0, 2.0])
    assert monitor.solves == [("Dense", True, None, None, None, None)]
