from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from stark.core.contracts.field import FieldLike
from stark.core.contracts.frame import FrameLike
from stark.core.contracts.norm import NormLike
from stark.core.contracts.translation import TranslationTypeContravariant


AlgebraistNormKernel = Callable[[TranslationTypeContravariant], float]


def included_norm_entries(
    frame: FrameLike,
) -> tuple[tuple[FieldLike[Any, Any], NormLike[Any]], ...]:
    """Return field/norm pairs that participate in norm calculations."""

    return tuple(
        (field, norm)
        for field, norm in zip(frame.fields, frame.norms, strict=True)
        if getattr(norm, "kind", None) != "excluded"
    )


class AlgebraistNorm(Protocol[TranslationTypeContravariant]):
    """Provider of frame-aware translation norm kernels."""

    def provide(self, request: None = None) -> AlgebraistNormKernel[TranslationTypeContravariant]:
        ...


__all__ = [
    "AlgebraistNorm",
    "AlgebraistNormKernel",
    "included_norm_entries",
]
