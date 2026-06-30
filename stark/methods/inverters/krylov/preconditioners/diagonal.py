from __future__ import annotations

from typing import Generic, Protocol, cast

from stark.core.contracts import (
    BlockLike,
    BlockOperatorDiagonalLike,
    BlockOperatorLike,
    TranslationType,
    TranslationTypeContravariant,
)


class InverterPreconditionerInvertibleEntry(Protocol[TranslationTypeContravariant]):
    """Diagonal operator entry that exposes an inverse action."""

    def inverse(
        self,
        source: TranslationTypeContravariant,
        target: TranslationTypeContravariant,
    ) -> None:
        ...


class InverterPreconditionerDiagonalInverse(Generic[TranslationType]):
    """Block-diagonal preconditioner using entry-level inverse actions.

    This preconditioner is intentionally narrow. It expects the operator to be
    indexable by block entry and each diagonal entry to provide
    ``inverse(source_entry, target_entry)``. That matches diagonal block
    operators and makes the requirement explicit instead of hiding it behind a
    generic callable.
    """

    __slots__ = ()

    def __call__(
        self,
        operator: BlockOperatorLike[TranslationType],
        source: BlockLike[TranslationType],
        target: BlockLike[TranslationType],
    ) -> None:
        diagonal = cast(BlockOperatorDiagonalLike[TranslationType], operator)
        for index in range(len(source)):
            entry = diagonal[index]
            if entry is None:
                raise RuntimeError(
                    f"Krylov diagonal preconditioner entry {index} is not configured."
                )
            invertible = cast(InverterPreconditionerInvertibleEntry[TranslationType], entry)
            invertible.inverse(source[index], target[index])


__all__ = ["InverterPreconditionerDiagonalInverse"]
