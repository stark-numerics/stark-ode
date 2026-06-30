from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from stark.engines.shared.algebraist.frame import AlgebraistFrame
from stark.engines.jax.carriers import CarrierJax


@dataclass
class EngineTranslationJax:
    """Structured JAX translation using return-style field carrier arithmetic."""

    algebraist_frame: AlgebraistFrame = field(repr=False)
    carriers: tuple[CarrierJax, ...] = field(repr=False)
    allocator: Any = field(repr=False)
    linear_combine: tuple[Any, ...] = field(default=(), repr=False)
    apply_translation: Any = field(default=None, repr=False)
    norm_kernel: Any = field(default=None, repr=False)

    def __call__(self, origin: object, result: object) -> None:
        if self.apply_translation is not None:
            self.apply_translation(origin, self, result)
            return

        for field, carrier in zip(self.algebraist_frame.fields, self.carriers, strict=True):
            field.state_path.assign(
                result,
                carrier.arithmetic.translate(
                    field.state_path(origin),
                    1.0,
                    field.translation_path(self),
                    field.state_path(result),
                ),
            )

    def __add__(self, other: EngineTranslationJax) -> EngineTranslationJax:
        if self.allocator is not other.allocator:
            raise ValueError("Cannot add translations allocated by different engines.")

        result = self.allocator.allocate_translation()
        for field, carrier in zip(self.algebraist_frame.fields, self.carriers, strict=True):
            field.translation_path.assign(
                result,
                carrier.arithmetic.add(
                    field.translation_path(self),
                    field.translation_path(other),
                    field.translation_path(result),
                ),
            )
        return result

    def __rmul__(self, scalar: float) -> EngineTranslationJax:
        result = self.allocator.allocate_translation()
        for field, carrier in zip(self.algebraist_frame.fields, self.carriers, strict=True):
            field.translation_path.assign(
                result,
                carrier.arithmetic.scale(
                    scalar,
                    field.translation_path(self),
                    field.translation_path(result),
                ),
            )
        return result

    def __mul__(self, scalar: float) -> EngineTranslationJax:
        return self.__rmul__(scalar)

    def norm(self) -> float:
        if self.norm_kernel is not None:
            return float(self.norm_kernel(self))

        total = 0.0
        for field, carrier in zip(self.algebraist_frame.fields, self.carriers, strict=True):
            if not field.norm.include:
                continue
            field_norm = carrier.norm(field.translation_path(self))
            total += field_norm * field_norm
        return sqrt(total)


__all__ = ["EngineTranslationJax"]
