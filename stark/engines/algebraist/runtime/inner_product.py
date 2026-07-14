from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from stark.core.contracts.frame import FrameLike
from stark.engines.algebraist.inner_product import included_inner_product_entries

TranslationType = TypeVar("TranslationType")


@dataclass(frozen=True, slots=True)
class AlgebraistRuntimeInnerProduct(Generic[TranslationType]):
    """Runtime provider of frame-aware translation inner products."""

    frame: FrameLike

    def provide(self, request: None = None) -> Callable[[TranslationType, TranslationType], float]:
        del request
        entries = included_inner_product_entries(self.frame)

        def kernel(left: TranslationType, right: TranslationType) -> float:
            total = 0.0
            for field, inner_product in entries:
                left_value = field.translation_path(left)
                right_value = field.translation_path(right)
                total += inner_product(left_value, right_value)
            return total

        return kernel


__all__ = ["AlgebraistRuntimeInnerProduct"]
