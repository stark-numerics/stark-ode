from __future__ import annotations

from typing import Protocol, TypeVar

BlockEntryType = TypeVar("BlockEntryType")
"""Type variable for entries stored in a block container.

Most method-level blocks contain full STARK translations, but coordinate
materialisation also uses blocks of backend carrier values. The container
contract is therefore intentionally unbound; contracts that need translation
algebra should keep using `TranslationType`.
"""

BlockEntryTypeContravariant = TypeVar("BlockEntryTypeContravariant", contravariant=True)
"""Contravariant type variable for callables that consume block entries."""


class BlockLike(Protocol[BlockEntryType]):
    """
    Structural contract for a block of translations.

    A block is an ordered product-space container used by numerical methods to
    group related translations. The entries may represent stage increments,
    coupled residual components, partitioned state variables, or other
    method-level collections.

    The contract describes only the shape required by block-level algorithms:
    entries can be inspected and replaced by index, the block has a stable
    length, and the whole block can be overwritten from another compatible
    block.

    Implementations may choose their own storage and arithmetic strategy. This
    protocol does not imply that block entries are independent, nor that a
    block operator acting on the block is diagonal.
    """
    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> BlockEntryType:
        ...

    def __setitem__(self, index: int, value: BlockEntryType) -> None:
        ...

    def replace(self, other: "BlockLike[BlockEntryType]", /) -> None:
        ...

    def norm(self) -> float:
        ...


class BlockOperatorLike(Protocol[BlockEntryType]):
    """
    General block operator.

    Represents any block-level linear action:

        target <- operator(source)

    The operator may couple entries of the source block.
    """

    def __call__(
        self,
        source: BlockLike[BlockEntryType],
        target: BlockLike[BlockEntryType],
        /,
    ) -> BlockLike[BlockEntryType]:
        ...


class BlockOperatorDiagonalLike(Protocol[BlockEntryType]):
    """
    Diagonal block operator.

    Represents an entrywise block-level linear action:

        target[i] <- operator[i](source[i])

    The diagonal entries are inspectable through integer indexing.
    """

    def __len__(self) -> int:
        ...

    def __getitem__(
        self,
        index: int,
    ) -> BlockOperatorEntryLike[BlockEntryType] | None:
        ...

    def __call__(
        self,
        source: BlockLike[BlockEntryType],
        target: BlockLike[BlockEntryType],
        /,
    ) -> BlockLike[BlockEntryType]:
        ...


class BlockOperatorEntryLike(Protocol[BlockEntryTypeContravariant]):
    """
    Structural contract for a linear operator on one block entry.

    A block entry operator acts on a single translation and writes its image
    into a supplied output translation. Diagonal block operators use these
    entry operators to represent entrywise actions of the form

        target[i] <- operator[i](source[i])

    This contract describes only the local translation-level action. It does
    not imply anything about block-level coupling; coupled block operators may
    exist without exposing entry operators.
    """
    def __call__(
        self,
        source: BlockEntryTypeContravariant,
        target: BlockEntryTypeContravariant,
        /,
    ) -> None:
        ...
