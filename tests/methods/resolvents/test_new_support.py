from __future__ import annotations

from typing import Any

from stark.core.block import Block, BlockSpecialist
from stark.methods.resolvents.specialization.stencil import ResolventStencilBlock
from tests.support import DummyScalarTranslation


class ItemSpecialist:
    def provide_delta(self, stencil: ResolventStencilBlock):
        def kernel(step: float, *items: Any) -> DummyScalarTranslation:
            sources = items[:-1]
            out = items[-1]
            out.value = step * stencil.scale * sum(
                coefficient * item.value
                for coefficient, item in zip(stencil.coefficients, sources)
            )
            return out

        return kernel

    def provide_apply(self, stencil: ResolventStencilBlock):
        del stencil
        raise NotImplementedError("This fixture only provides delta kernels.")


def test_resolvent_stencil_block_normalizes_coefficients() -> None:
    stencil = ResolventStencilBlock([1, -2], scale=0.5)

    assert stencil.coefficients == (1.0, -2.0)
    assert stencil.scale == 0.5
    assert stencil.apply is False


def test_block_specialist_uplifts_entry_kernel() -> None:
    specialist = BlockSpecialist(ItemSpecialist())
    kernel = specialist.provide(ResolventStencilBlock((1.0, -1.0)))

    out = Block([DummyScalarTranslation(0.0), DummyScalarTranslation(0.0)])
    left = Block([DummyScalarTranslation(3.0), DummyScalarTranslation(5.0)])
    right = Block([DummyScalarTranslation(1.0), DummyScalarTranslation(2.0)])

    result = kernel(1.0, left, right, out)

    assert result is out
    assert [item.value for item in out] == [2.0, 3.0]
