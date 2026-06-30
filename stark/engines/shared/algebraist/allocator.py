from __future__ import annotations

from typing import Protocol

from stark.core.contracts.translation import TranslationTypeCovariant


class AlgebraistAllocator(Protocol[TranslationTypeCovariant]):
    """Minimal allocator surface needed by Algebraist runtime providers."""

    def allocate_translation(self) -> TranslationTypeCovariant:
        ...
