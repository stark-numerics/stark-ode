from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import Generic, TypeVar

from stark.engines.shared.algebraist.frame import (
    AlgebraistFrame,
    AlgebraistFrameNormMax,
    AlgebraistFrameNormRMS,
)

TranslationType = TypeVar("TranslationType")


@dataclass(frozen=True, slots=True)
class AlgebraistRuntimeInnerProduct(Generic[TranslationType]):
    """Runtime provider of frame-aware translation inner products."""

    frame: AlgebraistFrame

    def provide(self, request: None = None) -> Callable[[TranslationType, TranslationType], float]:
        del request
        fields = self.frame.fields

        def kernel(left: TranslationType, right: TranslationType) -> float:
            total = 0.0
            for field in fields:
                if not field.norm.include:
                    continue
                scale = 1.0
                if isinstance(field.norm, AlgebraistFrameNormRMS):
                    scale = float(prod(field.policy.shape))
                elif not isinstance(field.norm, AlgebraistFrameNormMax):
                    raise ValueError("Runtime inner product requires RMS or max norm fields.")
                left_value = field.translation_path.get(left)
                right_value = field.translation_path.get(right)
                total += float((left_value * right_value).sum()) / scale
            return total

        return kernel


__all__ = ["AlgebraistRuntimeInnerProduct"]
