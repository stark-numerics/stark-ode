from __future__ import annotations

from stark.core.block import Block, BlockLinearFixed
from stark.methods.schemes.specialization.stencil import SchemeStencil
from tests.support import DummyScalarTranslation


def block(*values: float) -> Block[DummyScalarTranslation]:
    return Block([DummyScalarTranslation(value) for value in values])


def values(block: Block[DummyScalarTranslation]) -> tuple[float, ...]:
    return tuple(item.value for item in block)


class DummyBlockItemLinearFixed:
    """Entry-kernel provider used to test block lifting.

    Block lifting works at translation-entry level, not scheme-state level, so
    the fixture intentionally accepts a translation as its apply origin.
    """

    def __call__(self, request: SchemeStencil):
        coefficients = tuple(request.coefficients)

        if request.apply:
            return self.apply_kernel(coefficients)
        return self.delta_kernel(coefficients)

    def delta_kernel(self, coefficients: tuple[float, ...]):
        def delta_kernel(
            step: float,
            *terms: DummyScalarTranslation,
        ) -> DummyScalarTranslation:
            *translations, out = terms
            total = 0.0
            for coefficient, translation in zip(coefficients, translations, strict=True):
                total += step * coefficient * translation.value
            out.value = total
            return out

        return delta_kernel

    def apply_kernel(self, coefficients: tuple[float, ...]):
        def apply_kernel(
            step: float,
            origin: DummyScalarTranslation,
            *terms: DummyScalarTranslation,
        ) -> DummyScalarTranslation:
            *translations, result = terms
            total = origin.value
            for coefficient, translation in zip(coefficients, translations, strict=True):
                total += step * coefficient * translation.value
            result.value = total
            return result

        return apply_kernel


def test_block_linear_fixed_lifts_delta_kernel_entrywise() -> None:
    linear_fixed = BlockLinearFixed(DummyBlockItemLinearFixed())
    kernel = linear_fixed(SchemeStencil((2.0, -1.0)))
    out = block(0.0, 0.0)

    result = kernel(0.5, block(10.0, 20.0), block(3.0, 4.0), out)

    assert result is out
    assert values(out) == (8.5, 18.0)


def test_block_linear_fixed_lifts_apply_kernel_entrywise() -> None:
    linear_fixed = BlockLinearFixed(DummyBlockItemLinearFixed())
    kernel = linear_fixed(SchemeStencil((2.0, -1.0), apply=True))
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
