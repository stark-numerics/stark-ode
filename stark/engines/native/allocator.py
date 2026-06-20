from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from stark.engines.algebraist.frame import AlgebraistFrame
from stark.engines.carriers.native import CarrierNativeArray
from stark.engines.native.translation import EngineTranslationNative


@dataclass(frozen=True, slots=True)
class EngineAllocatorNative:
    """Allocate structured native state and translation objects."""

    algebraist_frame: AlgebraistFrame
    carriers: tuple[CarrierNativeArray, ...] = field(repr=False)
    apply_translation: Any = field(default=None, repr=False)
    norm: Any = field(default=None, repr=False)
    inner_product: Any = field(default=None, repr=False)

    def allocate_state(self) -> SimpleNamespace:
        state = SimpleNamespace()
        for field, carrier in zip(self.algebraist_frame.fields, self.carriers, strict=True):
            field.state_path.set(state, carrier.allocation.zero_state())
        return state

    def copy_state(self, source: object, out: object) -> object:
        for field, carrier in zip(self.algebraist_frame.fields, self.carriers, strict=True):
            field.state_path.set(
                out,
                carrier.allocation.copy_state(field.state_path.get(source)),
            )
        return out

    def allocate_translation(self) -> EngineTranslationNative:
        translation = EngineTranslationNative(
            algebraist_frame=self.algebraist_frame,
            carriers=self.carriers,
            allocator=self,
            apply_translation=self.apply_translation,
            norm_kernel=self.norm,
        )
        for field, carrier in zip(self.algebraist_frame.fields, self.carriers, strict=True):
            field.translation_path.set(translation, carrier.allocation.zero_translation())
        return translation


__all__ = ["EngineAllocatorNative"]
