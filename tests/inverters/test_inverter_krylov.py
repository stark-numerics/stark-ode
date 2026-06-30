from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark import Configuration, Tolerance
from stark.core.block import Block
from stark.core.block.operator import BlockOperatorDiagonal
from stark.core.contracts import InverterRequest
from stark.methods.inverters.krylov import (
    InverterPreconditionerDiagonalInverse,
    InverterPreconditionerNone,
    InverterKrylovArnoldi,
)


@dataclass(slots=True)
class ScalarTranslation:
    value: float = 0.0

    def __call__(self, origin: object, result: object) -> None:
        del origin, result

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: ScalarTranslation) -> ScalarTranslation:
        return ScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> ScalarTranslation:
        return ScalarTranslation(scalar * self.value)


class ScalarAllocator:
    def allocate_state(self) -> object:
        return object()

    def copy_state(self, source: object, out: object) -> None:
        del source, out

    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


def scalar_inner_product(left: ScalarTranslation, right: ScalarTranslation) -> float:
    return left.value * right.value


@dataclass(slots=True)
class Request:
    operator: object
    residual: Block[ScalarTranslation]


def strict_configuration() -> Configuration:
    return Configuration(
        inverter_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        inverter_maximum_steps=8,
    )


def test_krylov_arnoldi_solves_scalar_linear_system() -> None:
    def scale_by_two(source: Block[ScalarTranslation], target: Block[ScalarTranslation]) -> None:
        target[0].value = 2.0 * source[0].value

    inverter = InverterKrylovArnoldi(
        ScalarAllocator(),
        scalar_inner_product,
        restart=4,
        configuration=strict_configuration(),
    )
    output = Block([ScalarTranslation(0.0)])
    request = Request(scale_by_two, Block([ScalarTranslation(4.0)]))

    inverter(request, output)  # type: ignore[arg-type]

    assert output[0].value == pytest.approx(2.0, abs=1.0e-10)


def test_krylov_arnoldi_improves_nonzero_initial_guess() -> None:
    def scale_by_two(source: Block[ScalarTranslation], target: Block[ScalarTranslation]) -> None:
        target[0].value = 2.0 * source[0].value

    inverter = InverterKrylovArnoldi(
        ScalarAllocator(),
        scalar_inner_product,
        restart=4,
        configuration=strict_configuration(),
    )
    output = Block([ScalarTranslation(10.0)])
    request = Request(scale_by_two, Block([ScalarTranslation(4.0)]))

    inverter(request, output)  # type: ignore[arg-type]

    assert output[0].value == pytest.approx(2.0, abs=1.0e-10)


def test_krylov_arnoldi_solves_two_by_two_block_system() -> None:
    def apply_operator(source: Block[ScalarTranslation], target: Block[ScalarTranslation]) -> None:
        target[0].value = 4.0 * source[0].value + source[1].value
        target[1].value = source[0].value + 3.0 * source[1].value

    inverter = InverterKrylovArnoldi(
        ScalarAllocator(),
        scalar_inner_product,
        restart=4,
        configuration=strict_configuration(),
    )
    output = Block([ScalarTranslation(0.0), ScalarTranslation(0.0)])
    request = Request(
        apply_operator,
        Block([ScalarTranslation(1.0), ScalarTranslation(2.0)]),
    )

    inverter(request, output)  # type: ignore[arg-type]

    assert output[0].value == pytest.approx(1.0 / 11.0, abs=1.0e-10)
    assert output[1].value == pytest.approx(7.0 / 11.0, abs=1.0e-10)


def test_krylov_arnoldi_instance_uses_current_operator_contract() -> None:
    calls = 0

    def scale_by_two(source: Block[ScalarTranslation], target: Block[ScalarTranslation]) -> None:
        nonlocal calls
        calls += 1
        target[0].value = 2.0 * source[0].value

    inverter = InverterKrylovArnoldi(
        ScalarAllocator(),
        scalar_inner_product,
        restart=4,
        configuration=strict_configuration(),
    )
    solve = inverter.instance(scale_by_two)
    output = Block([ScalarTranslation(0.0)])

    solve(Block([ScalarTranslation(4.0)]), output)

    assert calls > 0
    assert output[0].value == pytest.approx(2.0, abs=1.0e-10)


def accepts_request(request: InverterRequest[ScalarTranslation]) -> float:
    return request.residual.norm()


def test_krylov_request_shape_remains_structural() -> None:
    def identity(source: Block[ScalarTranslation], target: Block[ScalarTranslation]) -> None:
        target[0].value = source[0].value

    request = Request(identity, Block([ScalarTranslation(3.0)]))

    assert accepts_request(request) == pytest.approx(3.0)


def test_inverter_preconditioner_none_uses_inverter_copy_semantics() -> None:
    copy_calls = 0

    def copy_block(
        source: Block[ScalarTranslation],
        target: Block[ScalarTranslation],
    ) -> None:
        nonlocal copy_calls
        copy_calls += 1
        for index in range(len(source)):
            target[index].value = source[index].value

    preconditioner = InverterPreconditionerNone(copy_block)
    source = Block([ScalarTranslation(2.0), ScalarTranslation(3.0)])
    target = Block([ScalarTranslation(0.0), ScalarTranslation(0.0)])

    preconditioner(object(), source, target)  # type: ignore[arg-type]

    assert copy_calls == 1
    assert target[0].value == pytest.approx(2.0)
    assert target[1].value == pytest.approx(3.0)
    assert target[0] is not source[0]


def test_inverter_preconditioner_diagonal_inverse_uses_entry_inverse_actions() -> None:
    @dataclass(slots=True)
    class ScaleEntry:
        scale: float

        def __call__(
            self,
            source: ScalarTranslation,
            target: ScalarTranslation,
        ) -> None:
            target.value = self.scale * source.value

        def inverse(
            self,
            source: ScalarTranslation,
            target: ScalarTranslation,
        ) -> None:
            target.value = source.value / self.scale

    operator = BlockOperatorDiagonal([ScaleEntry(2.0), ScaleEntry(4.0)])
    source = Block([ScalarTranslation(6.0), ScalarTranslation(8.0)])
    target = Block([ScalarTranslation(0.0), ScalarTranslation(0.0)])

    InverterPreconditionerDiagonalInverse()(operator, source, target)

    assert target[0].value == pytest.approx(3.0)
    assert target[1].value == pytest.approx(2.0)


def test_krylov_arnoldi_uses_left_preconditioner() -> None:
    def scale_by_two(source: Block[ScalarTranslation], target: Block[ScalarTranslation]) -> None:
        target[0].value = 2.0 * source[0].value

    class HalfPreconditioner:
        calls = 0

        def __call__(
            self,
            operator: object,
            source: Block[ScalarTranslation],
            target: Block[ScalarTranslation],
        ) -> None:
            del operator
            self.calls += 1
            target[0].value = 0.5 * source[0].value

    preconditioner = HalfPreconditioner()
    inverter = InverterKrylovArnoldi(
        ScalarAllocator(),
        scalar_inner_product,
        restart=4,
        configuration=strict_configuration(),
        preconditioner=preconditioner,
    )
    output = Block([ScalarTranslation(0.0)])
    request = Request(scale_by_two, Block([ScalarTranslation(4.0)]))

    inverter(request, output)  # type: ignore[arg-type]

    assert preconditioner.calls > 0
    assert output[0].value == pytest.approx(2.0, abs=1.0e-10)
