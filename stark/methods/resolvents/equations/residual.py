from __future__ import annotations

from typing import Protocol, TypeVar

from stark.core.contracts import BlockLike, TranslationType

TranslationOperatorTypeContravariant = TypeVar(
    "TranslationOperatorTypeContravariant",
    contravariant=True,
)
"""Contravariant output-operator type used by a residual differential.

Residual implementations often know a more specific differential object than
the package-wide operator contract. The type variable lets the residual mutate
that concrete object while the translation buffers stay type-preserving.
"""


class ResolventResidual(Protocol[TranslationType, TranslationOperatorTypeContravariant]):
    """Residual object used by direct resolvent implementations.

    The residual reads a candidate correction block and writes the residual
    block in-place. The translation type is carried through both blocks so
    method tests and user extensions can use precise scalar, vector, or
    backend-specific translation objects without losing type information.
    """

    def __call__(
        self,
        delta: BlockLike[TranslationType],
        out: BlockLike[TranslationType],
    ) -> None:
        ...

    def differential(
        self,
        delta: BlockLike[TranslationType],
        out: TranslationOperatorTypeContravariant,
    ) -> None:
        ...


__all__ = ["ResolventResidual"]
