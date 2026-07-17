"""Protocol for engine translation constructors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from stark.core.contracts.allocator import AllocatorLike
from stark.core.contracts.carrier import CarrierLike
from stark.core.contracts.frame import FrameLike
from stark.core.contracts.translation import TranslationType


class TranslationFactoryLike(Protocol[TranslationType]):
    """Construct a structured translation for a frame/carrier layout."""

    def __call__(
        self,
        *,
        frame: FrameLike,
        carriers: tuple[CarrierLike[Any, Any], ...],
        allocator: AllocatorLike[Any, TranslationType],
        linear_combine: tuple[Callable[..., TranslationType], ...] = (),
        apply_translation: Callable[[object, TranslationType, object], object] | None = None,
        norm_kernel: Callable[[TranslationType], float] | None = None,
    ) -> TranslationType:
        ...


__all__ = ["TranslationFactoryLike"]
