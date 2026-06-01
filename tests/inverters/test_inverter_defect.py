from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark.block import Block
from stark.block.operator import BlockOperatorDiagonal
from stark.inverters.support import InverterDefect
from stark.resolvents.requests.inverter import ResolventInverterRequest


@dataclass(slots=True)
class TranslationScalar:
    value: float = 0.0

    def __call__(self, origin, result) -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: "TranslationScalar") -> "TranslationScalar":
        return TranslationScalar(self.value + other.value)

    def __rmul__(self, scalar: float) -> "TranslationScalar":
        return TranslationScalar(scalar * self.value)


def scale_translation(factor: float):
    def apply(source: TranslationScalar, target: TranslationScalar) -> None:
        target.value = factor * source.value

    return apply


def test_inverter_defect_returns_norm_and_stores_defect_block() -> None:
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(scale_translation(2.0), size=1),
        residual=Block([TranslationScalar(5.0)]),
    )
    output = Block([TranslationScalar(1.0)])
    defect = InverterDefect[TranslationScalar]()

    defect_norm = defect(request, output)

    assert defect_norm == pytest.approx(3.0)
    assert defect.block is not None
    assert defect.block[0].value == pytest.approx(3.0)


def test_inverter_defect_reuses_scratch_for_same_output_size() -> None:
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(scale_translation(2.0), size=1),
        residual=Block([TranslationScalar(5.0)]),
    )
    output = Block([TranslationScalar(1.0)])
    defect = InverterDefect[TranslationScalar]()

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
        operator=BlockOperatorDiagonal.repeated(scale_translation(1.0), size=1),
        residual=Block([TranslationScalar(1.0)]),
    )
    output_one = Block([TranslationScalar(0.0)])
    defect = InverterDefect[TranslationScalar]()

    defect(request_one, output_one)
    image_one = defect.image
    block_one = defect.block

    request_two = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(scale_translation(1.0), size=2),
        residual=Block([TranslationScalar(1.0), TranslationScalar(2.0)]),
    )
    output_two = Block([TranslationScalar(0.0), TranslationScalar(0.0)])

    defect_norm = defect(request_two, output_two)

    assert defect.image is not image_one
    assert defect.block is not block_one
    assert defect.size == 2
    assert defect_norm == pytest.approx(5.0**0.5)
