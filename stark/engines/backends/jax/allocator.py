from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from stark.engines.algebraist.layout import AlgebraistLayout
from stark.engines.carriers.jax import CarrierJax
from stark.engines.backends.jax.translation import EngineTranslationJax


@dataclass(frozen=True, slots=True)
class EngineAllocatorJax:
    """Allocate structured JAX state and translation objects."""

    algebraist_layout: AlgebraistLayout
    carriers: tuple[CarrierJax, ...] = field(repr=False)
    norm: Any = field(default=None, repr=False)
    inner_product: Any = field(default=None, repr=False)

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

    def allocate_translation(self) -> EngineTranslationJax:
        translation = EngineTranslationJax(
            algebraist_layout=self.algebraist_layout,
            carriers=self.carriers,
            allocator=self,
            norm_kernel=self.norm,
        )
        for field, carrier in zip(self.algebraist_layout.fields, self.carriers, strict=True):
            field.translation_path.set(translation, carrier.allocation.zero_translation())
        return translation


__all__ = ["EngineAllocatorJax"]
