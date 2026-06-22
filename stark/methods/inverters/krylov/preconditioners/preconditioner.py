from __future__ import annotations

from typing import Protocol

from stark.core.contracts import BlockLike, BlockOperatorLike, TranslationType


class InverterKrylovPreconditionerLike(Protocol[TranslationType]):
    """Approximate inverse action used inside an Arnoldi Krylov window.

    A preconditioner receives the current operator, a source block, and a
    writable target block. It should write an approximation to ``A^{-1} source``
    into ``target`` without changing ``source``. Implementations may ignore the
    operator, inspect it, or cache operator-specific factors.
    """

    def __call__(
        self,
        operator: BlockOperatorLike[TranslationType],
        source: BlockLike[TranslationType],
        target: BlockLike[TranslationType],
    ) -> None:
        ...


__all__ = ["InverterKrylovPreconditionerLike"]
