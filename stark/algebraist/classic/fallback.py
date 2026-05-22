# stark/algebraist/fallback.py

from dataclasses import dataclass
from collections.abc import Callable

from stark.algebraist.classic.combine import AlgebraistCombineResolver



@dataclass(frozen=True)
class AlgebraistFallback:
    """Default runtime algebra strategy: no generated code."""

    def resolve_combine(self, translation, allocate_translation: Callable):
        return AlgebraistCombineResolver.from_translation(
            translation,
            allocate_translation,
            )