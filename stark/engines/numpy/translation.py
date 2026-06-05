from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from stark.algebraist.layout import AlgebraistLayout
from stark.carriers.numpy import CarrierNumpy


@dataclass
class StarkEngineTranslationNumpy:
    """Structured NumPy translation using in-place field carrier arithmetic."""

    algebraist_layout: AlgebraistLayout = field(repr=False)
    carriers: tuple[CarrierNumpy, ...] = field(repr=False)
    allocator: Any = field(repr=False)

    @property
    def linear_combine(self) -> tuple[Any, ...]:
        return ()

    def __call__(self, origin: object, result: object) -> None:
        for field, carrier in zip(self.algebraist_layout.fields, self.carriers, strict=True):
            carrier.arithmetic.translate(
                field.state_path.get(origin),
                1.0,
                field.translation_path.get(self),
                field.state_path.get(result),
            )

    def __add__(self, other: StarkEngineTranslationNumpy) -> StarkEngineTranslationNumpy:
        if self.allocator is not other.allocator:
            raise ValueError("Cannot add translations allocated by different engines.")

        result = self.allocator.allocate_translation()
        for field, carrier in zip(self.algebraist_layout.fields, self.carriers, strict=True):
            carrier.arithmetic.add(
                field.translation_path.get(self),
                field.translation_path.get(other),
                field.translation_path.get(result),
            )
        return result

    def __rmul__(self, scalar: float) -> StarkEngineTranslationNumpy:
        result = self.allocator.allocate_translation()
        for field, carrier in zip(self.algebraist_layout.fields, self.carriers, strict=True):
            carrier.arithmetic.scale(
                scalar,
                field.translation_path.get(self),
                field.translation_path.get(result),
            )
        return result

    def __mul__(self, scalar: float) -> StarkEngineTranslationNumpy:
        return self.__rmul__(scalar)

    def norm(self) -> float:
        total = 0.0
        for field, carrier in zip(self.algebraist_layout.fields, self.carriers, strict=True):
            if not field.include_in_norm:
                continue
            field_norm = carrier.norm(field.translation_path.get(self))
            total += field_norm * field_norm
        return sqrt(total)


__all__ = ["StarkEngineTranslationNumpy"]
