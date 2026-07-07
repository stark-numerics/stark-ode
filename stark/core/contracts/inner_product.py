"""Contracts for translation-space inner products."""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from stark.core.contracts.translation import TranslationFieldTypeContravariant


class InnerProduct(Protocol):
    """
    Return the inner product of two translations.

    Norms alone are not enough for Krylov methods. If a resolvent or inverter
    needs orthogonalization or secant projections, the user must also provide
    an inner product compatible with the translation space.
    """

    def __call__(
        self,
        left: Any,
        right: Any,
    ) -> float: ...


class InnerProductNamed(Protocol[TranslationFieldTypeContravariant]):
    """Frame-level inner product policy for one translation field."""

    kind: ClassVar[str]

    def __call__(
        self,
        left: TranslationFieldTypeContravariant,
        right: TranslationFieldTypeContravariant,
    ) -> float: ...


__all__ = ["InnerProduct", "InnerProductNamed"]
