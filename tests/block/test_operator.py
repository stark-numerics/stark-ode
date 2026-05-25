from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark.block import Block, BlockOperator


@dataclass
class TranslationFixture:
    value: float


def block(*values: float) -> Block[TranslationFixture]:
    return Block([TranslationFixture(value) for value in values])


def values(block: Block[TranslationFixture]) -> tuple[float, ...]:
    return tuple(item.value for item in block)


def add_one(translation: TranslationFixture, out: TranslationFixture) -> None:
    out.value = translation.value + 1.0


def double(translation: TranslationFixture, out: TranslationFixture) -> None:
    out.value = 2.0 * translation.value


def test_block_operator_applies_entrywise() -> None:
    operator = BlockOperator([add_one, double])
    out = block(0.0, 0.0)

    result = operator(block(3.0, 4.0), out)

    assert result is out
    assert values(out) == (4.0, 8.0)


def test_repeated_block_operator_reuses_operator_for_each_entry() -> None:
    operator = BlockOperator.repeated(add_one, 3)
    out = block(0.0, 0.0, 0.0)

    operator(block(1.0, 2.0, 3.0), out)

    assert values(out) == (2.0, 3.0, 4.0)


def test_block_operator_rejects_size_mismatch() -> None:
    operator = BlockOperator([add_one])

    with pytest.raises(ValueError, match="Block operator size"):
        operator(block(1.0, 2.0), block(0.0, 0.0))

    with pytest.raises(ValueError, match="Block sizes"):
        operator(block(1.0), block(0.0, 0.0))
