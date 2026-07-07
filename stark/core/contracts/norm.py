"""Contracts for norm policies."""

from __future__ import annotations

from typing import ClassVar, Protocol

from stark.core.contracts.translation import TranslationFieldTypeContravariant


class NormLike(Protocol[TranslationFieldTypeContravariant]):
    """Minimal norm policy contract understood by frame-aware engines."""

    kind: ClassVar[str]

    def __call__(self, translation_field: TranslationFieldTypeContravariant) -> float: ...


__all__ = ["NormLike"]
