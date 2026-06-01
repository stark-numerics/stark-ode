from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from stark.block.block import Block


TranslationType = TypeVar("TranslationType")
BlockEntryOperator = Callable[[TranslationType, TranslationType], None]


@dataclass(slots=True, init=False)
class BlockOperatorDiagonal(Generic[TranslationType]):
    """Entrywise linear operator on a Block."""

    operators: list[BlockEntryOperator[TranslationType] | None]

    def __init__(
        self,
        operators: Iterable[BlockEntryOperator[TranslationType] | None],
    ) -> None:
        self.operators = list(operators)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(operators={tuple(self.operators)!r})"

    def __str__(self) -> str:
        return f"block operator[{len(self.operators)}]"

    def __len__(self) -> int:
        return len(self.operators)

    def __getitem__(
        self,
        index: int,
    ) -> BlockEntryOperator[TranslationType] | None:
        return self.operators[index]

    def __setitem__(
        self,
        index: int,
        operator: BlockEntryOperator[TranslationType] | None,
    ) -> None:
        self.operators[index] = operator

    def reset(self) -> None:
        for index in range(len(self.operators)):
            self.operators[index] = None

    def __call__(
        self,
        block: Block[TranslationType],
        out: Block[TranslationType],
    ) -> Block[TranslationType]:
        Block._require_same_size(block, out)

        if len(block) != len(self.operators):
            raise ValueError(
                "Block operator size must match the input block size."
            )

        for index, operator in enumerate(self.operators):
            if operator is None:
                raise RuntimeError(
                    f"BlockOperatorDiagonal entry {index} is not configured."
                )

            operator(block[index], out[index])

        return out

    @classmethod
    def repeated(
        cls,
        operator: BlockEntryOperator[TranslationType],
        size: int,
    ) -> BlockOperatorDiagonal[TranslationType]:
        if type(size) is not int:
            raise TypeError("BlockOperatorDiagonal size must be an integer.")
        if size < 0:
            raise ValueError("BlockOperatorDiagonal size must be non-negative.")

        return cls(operator for _ in range(size))


__all__ = ["BlockEntryOperator", "BlockOperatorDiagonal"]
