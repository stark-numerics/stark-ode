from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark.block import BlockAllocator


@dataclass
class TranslationFixture:
    value: float = 0.0


class WorkbenchFixture:
    def __init__(self) -> None:
        self.count = 0

    def allocate_translation(self) -> TranslationFixture:
        self.count += 1
        return TranslationFixture(float(self.count))


def test_block_allocator_allocates_translation_entries() -> None:
    workbench = WorkbenchFixture()
    allocator = BlockAllocator(workbench)

    block = allocator.allocate(3)

    assert [entry.value for entry in block] == [1.0, 2.0, 3.0]
    assert workbench.count == 3


def test_block_allocator_allocates_like_existing_block() -> None:
    workbench = WorkbenchFixture()
    allocator = BlockAllocator(workbench)
    existing = allocator.allocate(2)

    result = allocator.allocate_like(existing)

    assert len(result) == 2
    assert [entry.value for entry in result] == [3.0, 4.0]


@pytest.mark.parametrize("size", [-1])
def test_block_allocator_rejects_negative_sizes(size: int) -> None:
    allocator = BlockAllocator(WorkbenchFixture())

    with pytest.raises(ValueError, match="non-negative"):
        allocator.allocate(size)


@pytest.mark.parametrize("size", [1.5, True, "2"])
def test_block_allocator_rejects_non_int_sizes(size: object) -> None:
    allocator = BlockAllocator(WorkbenchFixture())

    with pytest.raises(TypeError, match="int"):
        allocator.allocate(size)  # type: ignore[arg-type]
