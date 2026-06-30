from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic

from stark.core.contracts import BlockLike, BlockOperatorLike, TranslationType


@dataclass(slots=True)
class InverterPreconditionerNone(Generic[TranslationType]):
    """Identity preconditioner used when Arnoldi should run unpreconditioned.

    The copy operation is supplied by the owning inverter so the preconditioner
    keeps the same translation-copy semantics as the rest of the Krylov
    workspace. This avoids assuming that a ``Block`` entry replacement is a
    value copy.
    """

    copy_block: Callable[[BlockLike[TranslationType], BlockLike[TranslationType]], None]

    def __call__(
        self,
        operator: BlockOperatorLike[TranslationType],
        source: BlockLike[TranslationType],
        target: BlockLike[TranslationType],
    ) -> None:
        del operator
        self.copy_block(source, target)


__all__ = ["InverterPreconditionerNone"]
