from __future__ import annotations

import math

from stark.core.block import Block


class Translation:
    def __init__(self, value: float) -> None:
        self.value = value

    def __add__(self, other):
        return Translation(self.value + other.value)

    def __rmul__(self, scalar: float):
        return Translation(scalar * self.value)

    def norm(self) -> float:
        return abs(self.value)


def values(block: Block[Translation]) -> tuple[float, ...]:
    return tuple(item.value for item in block)


def test_block_replace_preserves_outer_block() -> None:
    block = Block([Translation(1.0), Translation(2.0)])
    replacement = Block([Translation(3.0), Translation(4.0)])

    block.replace(replacement)

    assert values(block) == (3.0, 4.0)


def test_block_inplace_subtracts_entrywise() -> None:
    block = Block([Translation(3.0), Translation(5.0)])
    residual = Block([Translation(1.0), Translation(2.0)])

    block -= residual

    assert values(block) == (2.0, 3.0)


def test_block_arithmetic_trusts_prepared_sizes() -> None:
    result = Block([Translation(1.0)]) + Block([Translation(1.0), Translation(2.0)])

    assert values(result) == (2.0,)


def test_block_norm_uses_product_norm() -> None:
    block = Block([Translation(3.0), Translation(4.0)])

    assert math.isclose(block.norm(), 5.0)
