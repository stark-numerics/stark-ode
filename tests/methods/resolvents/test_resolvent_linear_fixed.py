from __future__ import annotations

from typing import Any

from stark.core.block import Block, BlockLinearFixed
from stark.methods.resolvents.specialization.stencil import ResolventStencilBlock
from tests.support import DummyScalarTranslation


class DummyResolventItemLinearFixed:
    """Scalar item linear_fixed used to test block lifting for resolvent stencils."""

    def __call__(self, request: ResolventStencilBlock):
        if request.apply:
            raise NotImplementedError("This fixture only provides delta kernels.")

        def kernel(step: float, *items: Any) -> DummyScalarTranslation:
            sources = items[:-1]
            out = items[-1]
            out.value = step * request.scale * sum(
                coefficient * item.value
                for coefficient, item in zip(request.coefficients, sources)
            )
            return out

        return kernel


def test_resolvent_stencil_block_normalizes_coefficients() -> None:
    stencil = ResolventStencilBlock([1, -2], scale=0.5)

    assert stencil.coefficients == (1.0, -2.0)
    assert stencil.scale == 0.5
    assert stencil.apply is False


def test_block_linear_fixed_uplifts_entry_kernel() -> None:
    linear_fixed = BlockLinearFixed(DummyResolventItemLinearFixed())
    kernel = linear_fixed(ResolventStencilBlock((1.0, -1.0)))

    out = Block([DummyScalarTranslation(0.0), DummyScalarTranslation(0.0)])
    left = Block([DummyScalarTranslation(3.0), DummyScalarTranslation(5.0)])
    right = Block([DummyScalarTranslation(1.0), DummyScalarTranslation(2.0)])

    result = kernel(1.0, left, right, out)

    assert result is out
    assert [item.value for item in out] == [2.0, 3.0]
