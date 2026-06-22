from __future__ import annotations

from stark.core.contracts import BlockLike, BlockOperatorLike, TranslationType


class PreconditionerDiagonalInverse:
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
        for index in range(len(source)):
            operator[index].inverse(source[index], target[index])  # type: ignore[index, union-attr]


__all__ = ["PreconditionerDiagonalInverse"]
