"""Contracts for matrix-free linear operators."""

from __future__ import annotations

from typing import Protocol, TypeVar

from stark.core.contracts.translation import Translation, TranslationTypeContravariant


class Operator(Protocol[TranslationTypeContravariant]):
    """
    Fill `out` with the image of a translation under a linear operator.

    Users normally provide operators through a `LinearizerLike`. The operator does
    not need to expose a dense matrix. It only needs to apply the local linear
    map to a translation, which is enough for matrix-free inverters such as
    GMRES, FGMRES, and BiCGStab.

    Operators are generic in the translation type they consume. The type
    variable is contravariant because an operator is a worker that accepts input
    and output translations; it does not manufacture a more specific
    translation family.
    """

    def __call__(
        self,
        translation: TranslationTypeContravariant,
        out: TranslationTypeContravariant,
    ) -> None:
        """Write `operator(translation)` into `out`."""
        ...


OperatorType = TypeVar("OperatorType", bound=Operator)


__all__ = ["Operator", "OperatorType"]
