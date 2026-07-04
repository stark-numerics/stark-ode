from __future__ import annotations

import pytest

from stark.core.block import Block
from stark.core.block.operator import BlockOperatorDiagonal
from stark.methods.inverters.support import InverterDefect
from stark.methods.resolvents.requests.inverter import ResolventInverterRequest
from tests.support import DummyScalarEntryOperator, DummyScalarTranslation


def test_inverter_defect_returns_norm_and_stores_defect_block() -> None:
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(DummyScalarEntryOperator(2.0), size=1),
        residual=Block([DummyScalarTranslation(5.0)]),
    )
    output = Block([DummyScalarTranslation(1.0)])
    defect = InverterDefect[DummyScalarTranslation]()

    defect_norm = defect(request, output)

    assert defect_norm == pytest.approx(3.0)
    assert defect.block is not None
    assert defect.block[0].value == pytest.approx(3.0)


def test_inverter_defect_reuses_scratch_for_same_output_size() -> None:
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(DummyScalarEntryOperator(2.0), size=1),
        residual=Block([DummyScalarTranslation(5.0)]),
    )
    output = Block([DummyScalarTranslation(1.0)])
    defect = InverterDefect[DummyScalarTranslation]()

    defect(request, output)
    image = defect.image
    block = defect.block

    output[0].value = 2.0
    defect_norm = defect(request, output)

    assert defect.image is image
    assert defect.block is block
    assert defect_norm == pytest.approx(1.0)
    assert defect.block is not None
    assert defect.block[0].value == pytest.approx(1.0)


def test_inverter_defect_reallocates_scratch_when_output_size_changes() -> None:
    request_one = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(DummyScalarEntryOperator(1.0), size=1),
        residual=Block([DummyScalarTranslation(1.0)]),
    )
    output_one = Block([DummyScalarTranslation(0.0)])
    defect = InverterDefect[DummyScalarTranslation]()

    defect(request_one, output_one)
    image_one = defect.image
    block_one = defect.block

    request_two = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(DummyScalarEntryOperator(1.0), size=2),
        residual=Block([DummyScalarTranslation(1.0), DummyScalarTranslation(2.0)]),
    )
    output_two = Block([DummyScalarTranslation(0.0), DummyScalarTranslation(0.0)])

    defect_norm = defect(request_two, output_two)

    assert defect.image is not image_one
    assert defect.block is not block_one
    assert defect.size == 2
    assert defect_norm == pytest.approx(5.0**0.5)
