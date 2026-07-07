from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, SupportsFloat, cast

from stark.core.contracts.frame import FrameLike
from stark.engines.cupy.carriers import CarrierCupy


@dataclass
class EngineTranslationCupy:
    """Structured CuPy translation using in-place field carrier arithmetic."""

    algebraist_frame: FrameLike = field(repr=False)
    carriers: tuple[CarrierCupy, ...] = field(repr=False)
    allocator: Any = field(repr=False)
    linear_combine: tuple[Any, ...] = field(default=(), repr=False)
    apply_translation: Any = field(default=None, repr=False)
    norm_kernel: Any = field(default=None, repr=False)

    def __call__(self, origin: object, result: object) -> None:
        if self.apply_translation is not None:
            self.apply_translation(origin, self, result)
            return

        for field, carrier in zip(self.algebraist_frame.fields, self.carriers, strict=True):
            carrier.arithmetic.translate(
                field.state_path(origin),
                1.0,
                field.translation_path(self),
                field.state_path(result),
            )

    def __add__(self, other: EngineTranslationCupy) -> EngineTranslationCupy:
        if self.allocator is not other.allocator:
            raise ValueError("Cannot add translations allocated by different engines.")

        result = self.allocator.allocate_translation()
        for field, carrier in zip(self.algebraist_frame.fields, self.carriers, strict=True):
            carrier.arithmetic.add(
                field.translation_path(self),
                field.translation_path(other),
                field.translation_path(result),
            )
        return result

    def __rmul__(self, scalar: float) -> EngineTranslationCupy:
        result = self.allocator.allocate_translation()
        for field, carrier in zip(self.algebraist_frame.fields, self.carriers, strict=True):
            carrier.arithmetic.scale(
                scalar,
                field.translation_path(self),
                field.translation_path(result),
            )
        return result

    def __mul__(self, scalar: float) -> EngineTranslationCupy:
        return self.__rmul__(scalar)

    def norm(self) -> float:
        if self.norm_kernel is not None:
            value = self.norm_kernel(self)
            item = getattr(value, "item", None)
            if callable(item):
                value = item()
            return float(cast(SupportsFloat, value))

        total = 0.0
        for field, norm, carrier in zip(
            self.algebraist_frame.fields,
            self.algebraist_frame.norms,
            self.carriers,
            strict=True,
        ):
            if getattr(norm, "kind", None) == "excluded":
                continue
            field_norm = carrier.norm(field.translation_path(self))
            total += field_norm * field_norm
        return sqrt(total)


__all__ = ["EngineTranslationCupy"]
