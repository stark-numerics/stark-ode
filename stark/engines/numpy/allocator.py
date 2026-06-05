from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from stark.algebraist.layout import AlgebraistLayout
from stark.carriers.numpy import CarrierNumpy
from stark.engines.numpy.translation import StarkEngineTranslationNumpy


@dataclass(frozen=True, slots=True)
class StarkEngineAllocatorNumpy:
    """Allocate structured NumPy state and translation objects."""

    algebraist_layout: AlgebraistLayout
    carriers: tuple[CarrierNumpy, ...] = field(repr=False)

    def allocate_state(self) -> SimpleNamespace:
        state = SimpleNamespace()
        for field, carrier in zip(self.algebraist_layout.fields, self.carriers, strict=True):
            field.state_path.set(state, carrier.allocation.zero_state())
        return state

    def copy_state(self, source: object, out: object) -> object:
        for field, carrier in zip(self.algebraist_layout.fields, self.carriers, strict=True):
            field.state_path.set(
                out,
                carrier.allocation.copy_state(field.state_path.get(source)),
            )
        return out

    def allocate_translation(self) -> StarkEngineTranslationNumpy:
        translation = StarkEngineTranslationNumpy(
            algebraist_layout=self.algebraist_layout,
            carriers=self.carriers,
            allocator=self,
        )
        for field, carrier in zip(self.algebraist_layout.fields, self.carriers, strict=True):
            field.translation_path.set(translation, carrier.allocation.zero_translation())
        return translation


__all__ = ["StarkEngineAllocatorNumpy"]
