from __future__ import annotations

from stark.core.block import Block
from tests.support import DummyScalarTranslation


def block(*values: float) -> Block[DummyScalarTranslation]:
    return Block([DummyScalarTranslation(value) for value in values])


def values(block: Block[DummyScalarTranslation]) -> tuple[float, ...]:
    return tuple(item.value for item in block)


def test_block_exposes_collection_protocol() -> None:
    subject = block(1.0, 2.0)

    assert len(subject) == 2
    assert values(subject) == (1.0, 2.0)
    assert subject[0].value == 1.0
    assert str(subject) == "block[2]"
    assert repr(subject) == "Block(size=2)"


def test_block_allows_entry_replacement() -> None:
    subject = block(1.0, 2.0)

    subject[1] = DummyScalarTranslation(5.0)

    assert values(subject) == (1.0, 5.0)


def test_block_adds_entrywise() -> None:
    result = block(1.0, 2.0) + block(3.0, 4.0)

    assert values(result) == (4.0, 6.0)


def test_block_subtracts_entrywise_using_translation_linear_operations() -> None:
    result = block(5.0, 7.0) - block(2.0, 3.0)

    assert values(result) == (3.0, 4.0)


def test_block_scales_entrywise() -> None:
    result = 2.0 * block(1.0, -3.0)

    assert values(result) == (2.0, -6.0)


def test_block_norm_uses_product_space_norm() -> None:
    assert block(3.0, 4.0).norm() == 5.0


def test_empty_block_norm_is_zero() -> None:
    assert Block([]).norm() == 0.0


def test_block_arithmetic_is_a_lean_entrywise_internal_operation() -> None:
    assert values(block(1.0) + block(1.0, 2.0)) == (2.0,)
    assert values(block(5.0) - block(2.0, 3.0)) == (3.0,)
