from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from stark.core.block import Block, BlockSpecialist
from stark.methods.schemes.specialization.stencil import SchemeStencil


@dataclass
class TranslationFixture:
    value: float


def block(*values: float) -> Block[TranslationFixture]:
    return Block([TranslationFixture(value) for value in values])


def values(block: Block[TranslationFixture]) -> tuple[float, ...]:
    return tuple(item.value for item in block)


class SpecialistFixture:
    def provide_delta(self, stencil: SchemeStencil) -> Callable[..., TranslationFixture]:
        coefficients = stencil.coefficients
        fixed_scale = stencil.scale

        def delta_kernel(
            step: float,
            *terms: TranslationFixture,
        ) -> TranslationFixture:
            sources = terms[:-1]
            out = terms[-1]
            out.value = step * fixed_scale * sum(
                coefficient * source.value
                for coefficient, source in zip(coefficients, sources, strict=True)
            )
            return out

        return delta_kernel

    def provide_apply(self, stencil: SchemeStencil) -> Callable[..., TranslationFixture]:
        coefficients = stencil.coefficients
        fixed_scale = stencil.scale

        def apply_kernel(
            step: float,
            origin: TranslationFixture,
            *terms: TranslationFixture,
        ) -> TranslationFixture:
            sources = terms[:-1]
            result = terms[-1]
            result.value = origin.value + step * fixed_scale * sum(
                coefficient * source.value
                for coefficient, source in zip(coefficients, sources, strict=True)
            )
            return result

        return apply_kernel


def test_block_specialist_lifts_delta_kernel_entrywise() -> None:
    specialist = BlockSpecialist(SpecialistFixture())
    kernel = specialist.provide(SchemeStencil((2.0, -1.0)))
    out = block(0.0, 0.0)

    result = kernel(0.5, block(10.0, 20.0), block(3.0, 4.0), out)

    assert result is out
    assert values(out) == (8.5, 18.0)


def test_block_specialist_lifts_apply_kernel_entrywise() -> None:
    specialist = BlockSpecialist(SpecialistFixture())
    kernel = specialist.provide(SchemeStencil((2.0, -1.0), apply=True))
    result = block(0.0, 0.0)

    returned = kernel(
        0.5,
        block(100.0, 200.0),
        block(10.0, 20.0),
        block(3.0, 4.0),
        result,
    )

    assert returned is result
    assert values(result) == (108.5, 218.0)
