from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Generic, Iterator, Self

from stark.core.contracts.block import BlockLike
from stark.core.contracts.translation import TranslationType


@dataclass(slots=True)
class Block(Generic[TranslationType]):
    """Product-space collection of translation-like entries.

    A ``Block`` is intentionally not a full Translation. It is the
    product-space object used by resolvents and inverters when several
    translation unknowns have to move together.

    Native arithmetic is kept deliberately small so inline algorithms can
    read like mathematics without turning ``Block`` into a workspace or an
    Algebraist provider.
    """

    items: list[TranslationType]

    def __post_init__(self) -> None:
        self.items = list(self.items)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={len(self.items)})"

    def __str__(self) -> str:
        return f"block[{len(self.items)}]"

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[TranslationType]:
        return iter(self.items)

    def __getitem__(self, index: int) -> TranslationType:
        return self.items[index]

    def __setitem__(self, index: int, value: TranslationType) -> None:
        self.items[index] = value

    def replace(self, other: BlockLike[TranslationType]) -> None:
        """Replace this block's entries with another block's entries."""

        self.items[:] = [other[index] for index in range(len(other))]

    def __add__(self, other: Block[TranslationType]) -> Block[TranslationType]:
        return type(self)(
            [
                left + right  # type: ignore[operator]
                for left, right in zip(self, other)
            ]
        )

    def __sub__(self, other: Block[TranslationType]) -> Block[TranslationType]:
        return type(self)(
            [
                left + (-1.0 * right)  # type: ignore[operator]
                for left, right in zip(self, other)
            ]
        )

    def __rmul__(self, scalar: float) -> Block[TranslationType]:
        return type(self)(
            [
                scalar * item  # type: ignore[operator]
                for item in self
            ]
        )

    def __iadd__(self, other: Block[TranslationType]) -> Self:
        self.replace(self + other)
        return self

    def __isub__(self, other: Block[TranslationType]) -> Self:
        self.replace(self - other)
        return self

    def norm(self) -> float:
        if not self.items:
            return 0.0
        return sqrt(sum(item.norm() ** 2 for item in self.items))  # type: ignore[attr-defined]


__all__ = ["Block"]
