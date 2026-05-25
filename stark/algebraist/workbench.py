from __future__ import annotations

from typing import Protocol, TypeVar

TranslationType = TypeVar("TranslationType")


class AlgebraistWorkbench(Protocol[TranslationType]):
    """Minimal workbench surface needed by Algebraist runtime providers."""

    def allocate_translation(self) -> TranslationType:
        ...
