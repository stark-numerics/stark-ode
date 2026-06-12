from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest

from stark.core.block import Block, BlockSpecialist
from stark.core.block.operator import BlockOperatorDiagonal
from stark.methods.inverters.relaxation import (
    InverterRelaxationJacobi,
    InverterRelaxationRichardson,
    InverterRelaxationStencilUpdate,
)
from stark import Configuration, Tolerance
from stark.methods.resolvents.requests.inverter import ResolventInverterRequest


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


@dataclass(slots=True)
class ScaleEntryOperator:
    scale: float

    def __call__(self, source: TranslationScalar, target: TranslationScalar) -> None:
        target.value = self.scale * source.value

    def inverse(self, source: TranslationScalar, target: TranslationScalar) -> None:
        target.value = source.value / self.scale


class RecordingItemSpecialist:
    def __init__(self) -> None:
        self.stencils: list[InverterRelaxationStencilUpdate] = []
        self.calls = 0

    def provide(
        self,
        stencil: InverterRelaxationStencilUpdate,
    ) -> Callable[..., TranslationScalar]:
        self.stencils.append(stencil)

        def kernel(
            step: float,
            *terms: TranslationScalar,
        ) -> TranslationScalar:
            self.calls += 1
            sources = terms[:-1]
            result = terms[-1]
            result.value = step * stencil.scale * sum(
                coefficient * source.value
                for coefficient, source in zip(stencil.coefficients, sources, strict=True)
            )
            return result

        return kernel


def scale_by_two(source: TranslationScalar, target: TranslationScalar) -> None:
    target.value = 2.0 * source.value


def invert_entry(
    operator: ScaleEntryOperator,
    source: TranslationScalar,
    target: TranslationScalar,
) -> None:
    operator.inverse(source, target)


def test_richardson_uses_specialist_update_path() -> None:
    item_specialist = RecordingItemSpecialist()
    specialist = BlockSpecialist(item_specialist)
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(scale_by_two, size=1),
        residual=Block([TranslationScalar(6.0)]),
    )
    output = Block([TranslationScalar(0.0)])
    inverter = InverterRelaxationRichardson[TranslationScalar](
        damping=0.5,
        configuration=Configuration(inverter_tolerance=Tolerance(atol=1.0e-12, rtol=0.0), inverter_maximum_steps=4),
        specialist=specialist,
    )

    inverter(request, output)

    assert output[0].value == pytest.approx(3.0)
    assert len(item_specialist.stencils) == 1
    assert item_specialist.stencils[0] == InverterRelaxationStencilUpdate(0.5)
    assert item_specialist.calls == 1


def test_jacobi_uses_specialist_update_path_after_diagonal_inverse() -> None:
    item_specialist = RecordingItemSpecialist()
    specialist = BlockSpecialist(item_specialist)
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal([ScaleEntryOperator(2.0), ScaleEntryOperator(4.0)]),
        residual=Block([TranslationScalar(6.0), TranslationScalar(20.0)]),
    )
    output = Block([TranslationScalar(0.0), TranslationScalar(0.0)])
    inverter = InverterRelaxationJacobi[TranslationScalar](
        invert_entry,
        configuration=Configuration(inverter_tolerance=Tolerance(atol=1.0e-12, rtol=0.0), inverter_maximum_steps=2),
        specialist=specialist,
    )

    inverter(request, output)

    assert output[0].value == pytest.approx(3.0)
    assert output[1].value == pytest.approx(5.0)
    assert len(item_specialist.stencils) == 1
    assert item_specialist.stencils[0] == InverterRelaxationStencilUpdate(1.0)
    assert item_specialist.calls == 2
