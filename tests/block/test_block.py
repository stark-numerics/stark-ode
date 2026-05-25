from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark.block import Block


@dataclass
class TranslationFixture:
    value: float

    def __call__(self, origin: float, result: list[float]) -> None:
        result[0] = origin + self.value

    def __add__(self, other: "TranslationFixture") -> "TranslationFixture":
        return TranslationFixture(self.value + other.value)

    def __rmul__(self, scalar: float) -> "TranslationFixture":
        return TranslationFixture(scalar * self.value)

    def norm(self) -> float:
        return abs(self.value)


def block(*values: float) -> Block[TranslationFixture]:
    return Block([TranslationFixture(value) for value in values])


def values(block: Block[TranslationFixture]) -> tuple[float, ...]:
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

    subject[1] = TranslationFixture(5.0)

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


def test_block_arithmetic_rejects_mismatched_sizes() -> None:
    with pytest.raises(ValueError, match="Block sizes must match"):
        block(1.0) + block(1.0, 2.0)

    with pytest.raises(ValueError, match="Block sizes must match"):
        block(1.0) - block(1.0, 2.0)
