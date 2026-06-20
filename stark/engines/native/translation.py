from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from stark.engines.shared.algebraist.frame import AlgebraistFrame
from stark.engines.native.carriers import CarrierNativeArray


@dataclass
class EngineTranslationNative:
    """Structured native translation using in-place array carrier arithmetic."""

    algebraist_frame: AlgebraistFrame = field(repr=False)
    carriers: tuple[CarrierNativeArray, ...] = field(repr=False)
    allocator: Any = field(repr=False)
    apply_translation: Any = field(default=None, repr=False)
    norm_kernel: Any = field(default=None, repr=False)

    @property
    def linear_combine(self) -> tuple[Any, ...]:
        return ()

    def __call__(self, origin: object, result: object) -> None:
        if self.apply_translation is not None:
            self.apply_translation(origin, self, result)
            return

        for field, carrier in zip(self.algebraist_frame.fields, self.carriers, strict=True):
            carrier.arithmetic.translate(
                field.state_path.get(origin),
                1.0,
                field.translation_path.get(self),
                field.state_path.get(result),
            )

    def __add__(self, other: EngineTranslationNative) -> EngineTranslationNative:
        if self.allocator is not other.allocator:
            raise ValueError("Cannot add translations allocated by different engines.")

        result = self.allocator.allocate_translation()
        for field, carrier in zip(self.algebraist_frame.fields, self.carriers, strict=True):
            carrier.arithmetic.add(
                field.translation_path.get(self),
                field.translation_path.get(other),
                field.translation_path.get(result),
            )
        return result

    def __rmul__(self, scalar: float) -> EngineTranslationNative:
        result = self.allocator.allocate_translation()
        for field, carrier in zip(self.algebraist_frame.fields, self.carriers, strict=True):
            carrier.arithmetic.scale(
                scalar,
                field.translation_path.get(self),
                field.translation_path.get(result),
            )
        return result

    def __mul__(self, scalar: float) -> EngineTranslationNative:
        return self.__rmul__(scalar)

    def norm(self) -> float:
        if self.norm_kernel is not None:
            return float(self.norm_kernel(self))

        total = 0.0
        for field, carrier in zip(self.algebraist_frame.fields, self.carriers, strict=True):
            if not field.norm.include:
                continue
            field_norm = carrier.norm(field.translation_path.get(self))
            total += field_norm * field_norm
        return sqrt(total)


__all__ = ["EngineTranslationNative"]
