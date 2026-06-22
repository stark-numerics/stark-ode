from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from stark.core.contracts.block import BlockLike


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
        source: BlockLike[TranslationType],
        target: BlockLike[TranslationType],
    ) -> BlockLike[TranslationType]:
        for index, operator in enumerate(self.operators):
            operator(source[index], target[index])  # type: ignore[misc]

        return target

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
