from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark import Configuration, Tolerance
from stark.core.block import Block
from stark.core.block.operator import BlockOperatorDiagonal
from stark.core.contracts import BlockLike, BlockOperatorLike, InverterRequest
from stark.methods.inverters.krylov import (
    InverterPreconditionerDiagonalInverse,
    InverterPreconditionerNone,
    InverterKrylovArnoldi,
)
from tests.support import (
    DummyScalarAllocator,
    DummyScalarEntryOperator,
    DummyScalarTranslation,
    dummy_scalar_inner_product,
)


@dataclass(slots=True)
class Request:
    operator: BlockOperatorLike[DummyScalarTranslation]
    residual: BlockLike[DummyScalarTranslation]


class DummyBlockOperator:
    """No-op block operator used where a test preconditioner ignores `operator`."""

    def __call__(
        self,
        source: BlockLike[DummyScalarTranslation],
        target: BlockLike[DummyScalarTranslation],
    ) -> BlockLike[DummyScalarTranslation]:
        for index in range(len(source)):
            target[index].value = source[index].value
        return target


def strict_configuration() -> Configuration:
    return Configuration(
        inverter_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        inverter_maximum_steps=8,
    )


def test_krylov_arnoldi_solves_scalar_linear_system() -> None:
    def scale_by_two(
        source: BlockLike[DummyScalarTranslation],
        target: BlockLike[DummyScalarTranslation],
    ) -> BlockLike[DummyScalarTranslation]:
        target[0].value = 2.0 * source[0].value
        return target

    inverter = InverterKrylovArnoldi(
        DummyScalarAllocator(),
        dummy_scalar_inner_product,
        restart=4,
        configuration=strict_configuration(),
    )
    output = Block([DummyScalarTranslation(0.0)])
    request = Request(scale_by_two, Block([DummyScalarTranslation(4.0)]))

    inverter(request, output)

    assert output[0].value == pytest.approx(2.0, abs=1.0e-10)


def test_krylov_arnoldi_improves_nonzero_initial_guess() -> None:
    def scale_by_two(
        source: BlockLike[DummyScalarTranslation],
        target: BlockLike[DummyScalarTranslation],
    ) -> BlockLike[DummyScalarTranslation]:
        target[0].value = 2.0 * source[0].value
        return target

    inverter = InverterKrylovArnoldi(
        DummyScalarAllocator(),
        dummy_scalar_inner_product,
        restart=4,
        configuration=strict_configuration(),
    )
    output = Block([DummyScalarTranslation(10.0)])
    request = Request(scale_by_two, Block([DummyScalarTranslation(4.0)]))

    inverter(request, output)

    assert output[0].value == pytest.approx(2.0, abs=1.0e-10)


def test_krylov_arnoldi_solves_two_by_two_block_system() -> None:
    def apply_operator(
        source: BlockLike[DummyScalarTranslation],
        target: BlockLike[DummyScalarTranslation],
    ) -> BlockLike[DummyScalarTranslation]:
        target[0].value = 4.0 * source[0].value + source[1].value
        target[1].value = source[0].value + 3.0 * source[1].value
        return target

    inverter = InverterKrylovArnoldi(
        DummyScalarAllocator(),
        dummy_scalar_inner_product,
        restart=4,
        configuration=strict_configuration(),
    )
    output = Block([DummyScalarTranslation(0.0), DummyScalarTranslation(0.0)])
    request = Request(
        apply_operator,
        Block([DummyScalarTranslation(1.0), DummyScalarTranslation(2.0)]),
    )

    inverter(request, output)

    assert output[0].value == pytest.approx(1.0 / 11.0, abs=1.0e-10)
    assert output[1].value == pytest.approx(7.0 / 11.0, abs=1.0e-10)


def test_krylov_arnoldi_instance_uses_current_operator_contract() -> None:
    calls = 0

    def scale_by_two(
        source: BlockLike[DummyScalarTranslation],
        target: BlockLike[DummyScalarTranslation],
    ) -> BlockLike[DummyScalarTranslation]:
        nonlocal calls
        calls += 1
        target[0].value = 2.0 * source[0].value
        return target

    inverter = InverterKrylovArnoldi(
        DummyScalarAllocator(),
        dummy_scalar_inner_product,
        restart=4,
        configuration=strict_configuration(),
    )
    solve = inverter.instance(scale_by_two)
    output = Block([DummyScalarTranslation(0.0)])

    solve(Block([DummyScalarTranslation(4.0)]), output)

    assert calls > 0
    assert output[0].value == pytest.approx(2.0, abs=1.0e-10)


def accepts_request(request: InverterRequest[DummyScalarTranslation]) -> float:
    return request.residual.norm()


def test_krylov_request_shape_remains_structural() -> None:
    def identity(
        source: BlockLike[DummyScalarTranslation],
        target: BlockLike[DummyScalarTranslation],
    ) -> BlockLike[DummyScalarTranslation]:
        target[0].value = source[0].value
        return target

    request = Request(identity, Block([DummyScalarTranslation(3.0)]))

    assert accepts_request(request) == pytest.approx(3.0)


def test_inverter_preconditioner_none_uses_inverter_copy_semantics() -> None:
    copy_calls = 0

    def copy_block(
        source: BlockLike[DummyScalarTranslation],
        target: BlockLike[DummyScalarTranslation],
    ) -> None:
        nonlocal copy_calls
        copy_calls += 1
        for index in range(len(source)):
            target[index].value = source[index].value

    preconditioner = InverterPreconditionerNone(copy_block)
    source = Block([DummyScalarTranslation(2.0), DummyScalarTranslation(3.0)])
    target = Block([DummyScalarTranslation(0.0), DummyScalarTranslation(0.0)])

    preconditioner(DummyBlockOperator(), source, target)

    assert copy_calls == 1
    assert target[0].value == pytest.approx(2.0)
    assert target[1].value == pytest.approx(3.0)
    assert target[0] is not source[0]


def test_inverter_preconditioner_diagonal_inverse_uses_entry_inverse_actions() -> None:
    operator = BlockOperatorDiagonal([DummyScalarEntryOperator(2.0), DummyScalarEntryOperator(4.0)])
    source = Block([DummyScalarTranslation(6.0), DummyScalarTranslation(8.0)])
    target = Block([DummyScalarTranslation(0.0), DummyScalarTranslation(0.0)])

    InverterPreconditionerDiagonalInverse()(operator, source, target)

    assert target[0].value == pytest.approx(3.0)
    assert target[1].value == pytest.approx(2.0)


def test_krylov_arnoldi_uses_left_preconditioner() -> None:
    def scale_by_two(
        source: BlockLike[DummyScalarTranslation],
        target: BlockLike[DummyScalarTranslation],
    ) -> BlockLike[DummyScalarTranslation]:
        target[0].value = 2.0 * source[0].value
        return target

    class HalfPreconditioner:
        calls = 0

        def __call__(
            self,
            operator: BlockOperatorLike[DummyScalarTranslation],
            source: BlockLike[DummyScalarTranslation],
            target: BlockLike[DummyScalarTranslation],
        ) -> None:
            del operator
            self.calls += 1
            target[0].value = 0.5 * source[0].value

    preconditioner = HalfPreconditioner()
    inverter = InverterKrylovArnoldi(
        DummyScalarAllocator(),
        dummy_scalar_inner_product,
        restart=4,
        configuration=strict_configuration(),
        preconditioner=preconditioner,
    )
    output = Block([DummyScalarTranslation(0.0)])
    request = Request(scale_by_two, Block([DummyScalarTranslation(4.0)]))

    inverter(request, output)

    assert preconditioner.calls > 0
    assert output[0].value == pytest.approx(2.0, abs=1.0e-10)
