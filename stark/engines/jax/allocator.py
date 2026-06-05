from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from stark.algebraist.layout import AlgebraistLayout
from stark.carriers.jax import CarrierJax
from stark.engines.jax.translation import StarkEngineTranslationJax


@dataclass(frozen=True, slots=True)
class StarkEngineAllocatorJax:
    """Allocate structured JAX state and translation objects."""

    algebraist_layout: AlgebraistLayout
    carriers: tuple[CarrierJax, ...] = field(repr=False)

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

    def allocate_translation(self) -> StarkEngineTranslationJax:
        translation = StarkEngineTranslationJax(
            algebraist_layout=self.algebraist_layout,
            carriers=self.carriers,
            allocator=self,
        )
        for field, carrier in zip(self.algebraist_layout.fields, self.carriers, strict=True):
            field.translation_path.set(translation, carrier.allocation.zero_translation())
        return translation


__all__ = ["StarkEngineAllocatorJax"]
