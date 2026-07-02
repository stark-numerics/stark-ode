from __future__ import annotations

from dataclasses import dataclass

from stark.core.block import Block, BlockOperatorDiagonal


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
    operator = BlockOperatorDiagonal([add_one, double])
    out = block(0.0, 0.0)

    result = operator(block(3.0, 4.0), out)

    assert result is out
    assert values(out) == (4.0, 8.0)


def test_repeated_block_operator_reuses_operator_for_each_entry() -> None:
    operator = BlockOperatorDiagonal.repeated(add_one, 3)
    out = block(0.0, 0.0, 0.0)

    operator(block(1.0, 2.0, 3.0), out)

    assert values(out) == (2.0, 3.0, 4.0)


def test_block_operator_call_is_a_lean_prepared_internal_operation() -> None:
    operator = BlockOperatorDiagonal([add_one])
    out = block(0.0, 99.0)

    operator(block(1.0, 2.0), out)

    assert values(out) == (2.0, 99.0)
