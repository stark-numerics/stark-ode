from __future__ import annotations

from dataclasses import dataclass

from stark.core.block import Block, BlockSpecialist
from stark.methods.resolvents.specialization.stencil import ResolventStencilBlock


class ItemSpecialist:
    def provide_delta(self, stencil):
        def kernel(step, *items):
            sources = items[:-1]
            out = items[-1]
            out.value = step * stencil.scale * sum(
                coefficient * item.value
                for coefficient, item in zip(stencil.coefficients, sources)
            )
            return out

        return kernel

    def provide_apply(self, stencil):
        raise NotImplementedError("This fixture only provides delta kernels.")


@dataclass
class Translation:
    value: float

    def __add__(self, other):
        return Translation(self.value + other.value)

    def __rmul__(self, scalar: float):
        return Translation(scalar * self.value)

    def norm(self) -> float:
        return abs(self.value)


def test_resolvent_stencil_block_normalizes_coefficients() -> None:
    stencil = ResolventStencilBlock([1, -2], scale=0.5)

    assert stencil.coefficients == (1.0, -2.0)
    assert stencil.scale == 0.5
    assert stencil.apply is False


def test_block_specialist_uplifts_entry_kernel() -> None:
    specialist = BlockSpecialist(ItemSpecialist())
    kernel = specialist.provide(ResolventStencilBlock((1.0, -1.0)))

    out = Block([Translation(0.0), Translation(0.0)])
    left = Block([Translation(3.0), Translation(5.0)])
    right = Block([Translation(1.0), Translation(2.0)])

    result = kernel(1.0, left, right, out)

    assert result is out
    assert [item.value for item in out] == [2.0, 3.0]
