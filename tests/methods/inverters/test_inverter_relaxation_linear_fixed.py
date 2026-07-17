from __future__ import annotations

from typing import Callable, cast

import pytest

from stark.core.block import Block, BlockLinearFixed
from stark.core.block.operator import BlockOperatorDiagonal
from stark.core.contracts import BlockOperatorEntryLike
from stark.methods.inverters.relaxation import (
    InverterRelaxationJacobi,
    InverterRelaxationRichardson,
    InverterRelaxationStencilUpdate,
)
from stark import Configuration, Tolerance
from stark.methods.resolvents.requests.inverter import ResolventInverterRequest
from tests.support import DummyScalarEntryOperator, DummyScalarTranslation


class RecordingItemLinearFixed:
    def __init__(self) -> None:
        self.stencils: list[InverterRelaxationStencilUpdate] = []
        self.calls = 0

    def __call__(
        self,
        request: InverterRelaxationStencilUpdate,
    ) -> Callable[..., DummyScalarTranslation]:
        if request.apply:
            raise NotImplementedError(
                "Relaxation update fixtures only provide delta kernels."
            )

        self.stencils.append(request)

        def kernel(
            step: float,
            *terms: DummyScalarTranslation,
        ) -> DummyScalarTranslation:
            self.calls += 1
            sources = terms[:-1]
            result = terms[-1]
            result.value = step * request.scale * sum(
                coefficient * source.value
                for coefficient, source in zip(request.coefficients, sources, strict=True)
            )
            return result

        return kernel


def scale_by_two(source: DummyScalarTranslation, target: DummyScalarTranslation) -> None:
    target.value = 2.0 * source.value


def invert_entry(
    operator: BlockOperatorEntryLike[DummyScalarTranslation],
    source: DummyScalarTranslation,
    target: DummyScalarTranslation,
) -> None:
    cast(DummyScalarEntryOperator, operator).inverse(source, target)


def test_richardson_uses_linear_fixed_update_path() -> None:
    item_linear_fixed = RecordingItemLinearFixed()
    linear_fixed = BlockLinearFixed(item_linear_fixed)
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(scale_by_two, size=1),
        residual=Block([DummyScalarTranslation(6.0)]),
    )
    output = Block([DummyScalarTranslation(0.0)])
    inverter = InverterRelaxationRichardson[DummyScalarTranslation](
        damping=0.5,
        configuration=Configuration(inverter_tolerance=Tolerance(atol=1.0e-12, rtol=0.0), inverter_maximum_steps=4),
        linear_fixed=linear_fixed,
    )

    inverter(request, output)

    assert output[0].value == pytest.approx(3.0)
    assert len(item_linear_fixed.stencils) == 1
    assert item_linear_fixed.stencils[0] == InverterRelaxationStencilUpdate(0.5)
    assert item_linear_fixed.calls == 1


def test_jacobi_uses_linear_fixed_update_path_after_diagonal_inverse() -> None:
    item_linear_fixed = RecordingItemLinearFixed()
    linear_fixed = BlockLinearFixed(item_linear_fixed)
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal([DummyScalarEntryOperator(2.0), DummyScalarEntryOperator(4.0)]),
        residual=Block([DummyScalarTranslation(6.0), DummyScalarTranslation(20.0)]),
    )
    output = Block([DummyScalarTranslation(0.0), DummyScalarTranslation(0.0)])
    inverter = InverterRelaxationJacobi[DummyScalarTranslation](
        invert_entry,
        configuration=Configuration(inverter_tolerance=Tolerance(atol=1.0e-12, rtol=0.0), inverter_maximum_steps=2),
        linear_fixed=linear_fixed,
    )

    inverter(request, output)

    assert output[0].value == pytest.approx(3.0)
    assert output[1].value == pytest.approx(5.0)
    assert len(item_linear_fixed.stencils) == 1
    assert item_linear_fixed.stencils[0] == InverterRelaxationStencilUpdate(1.0)
    assert item_linear_fixed.calls == 2
