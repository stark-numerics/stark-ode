from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from stark.block import Block, BlockSpecialist
from stark.schemes.support.stencil import SchemeStencil


@dataclass
class TranslationFixture:
    value: float


def block(*values: float) -> Block[TranslationFixture]:
    return Block([TranslationFixture(value) for value in values])


def values(block: Block[TranslationFixture]) -> tuple[float, ...]:
    return tuple(item.value for item in block)


class SpecialistFixture:
    def provide(self, stencil: SchemeStencil) -> Callable[..., TranslationFixture]:
        coefficients = stencil.coefficients
        fixed_scale = stencil.scale

        if stencil.apply:
            def apply_kernel(
                step: float,
                result: TranslationFixture,
                origin: TranslationFixture,
                *sources: TranslationFixture,
            ) -> TranslationFixture:
                result.value = origin.value + step * fixed_scale * sum(
                    coefficient * source.value
                    for coefficient, source in zip(coefficients, sources, strict=True)
                )
                return result

            return apply_kernel

        def delta_kernel(
            step: float,
            out: TranslationFixture,
            *sources: TranslationFixture,
        ) -> TranslationFixture:
            out.value = step * fixed_scale * sum(
                coefficient * source.value
                for coefficient, source in zip(coefficients, sources, strict=True)
            )
            return out

        return delta_kernel


def test_block_specialist_lifts_delta_kernel_entrywise() -> None:
    specialist = BlockSpecialist(SpecialistFixture())
    kernel = specialist.provide(SchemeStencil((2.0, -1.0)))
    out = block(0.0, 0.0)

    result = kernel(0.5, out, block(10.0, 20.0), block(3.0, 4.0))

    assert result is out
    assert values(out) == (8.5, 18.0)


def test_block_specialist_lifts_apply_kernel_entrywise() -> None:
    specialist = BlockSpecialist(SpecialistFixture())
    kernel = specialist.provide(SchemeStencil((2.0, -1.0), apply=True))
    result = block(0.0, 0.0)

    returned = kernel(
        0.5,
        result,
        block(100.0, 200.0),
        block(10.0, 20.0),
        block(3.0, 4.0),
    )

    assert returned is result
    assert values(result) == (108.5, 218.0)
