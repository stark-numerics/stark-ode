"""Protocol for engine translation constructors."""

from __future__ import annotations

from typing import Any, Protocol

from stark.core.contracts.allocator import AllocatorLike
from stark.core.contracts.frame import FrameLike
from stark.core.contracts.translation import TranslationTypeCovariant


class TranslationFactoryLike(Protocol[TranslationTypeCovariant]):
    """Construct a structured translation for a frame/carrier layout."""

    def __call__(
        self,
        *,
        frame: FrameLike,
        carriers: tuple[Any, ...],
        allocator: AllocatorLike[Any, Any],
        linear_combine: tuple[Any, ...] = (),
        apply_translation: Any = None,
        norm_kernel: Any = None,
    ) -> TranslationTypeCovariant:
        ...


__all__ = ["TranslationFactoryLike"]
