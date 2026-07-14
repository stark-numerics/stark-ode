from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from stark.core.contracts.carrier import CarrierLike
from stark.core.contracts.frame import FrameLike
from stark.core.contracts.translation_factory import TranslationFactoryLike


@dataclass(frozen=True, slots=True)
class Allocator:
    """Allocate structured state and translation objects from carriers."""

    frame: FrameLike
    carriers: tuple[CarrierLike[Any, Any], ...] = field(repr=False)
    translation_type: TranslationFactoryLike[Any] = field(repr=False)
    linear_combine: tuple[Any, ...] = field(default=(), repr=False)
    apply_translation: Any = field(default=None, repr=False)
    norm: Any = field(default=None, repr=False)
    inner_product: Any = field(default=None, repr=False)

    def allocate_state(self) -> SimpleNamespace:
        state = SimpleNamespace()
        for field, carrier in zip(self.frame.fields, self.carriers, strict=True):
            field.state_path.assign(state, carrier.allocation.zero_state())
        return state

    def copy_state(self, source: object, out: object) -> object:
        for field, carrier in zip(self.frame.fields, self.carriers, strict=True):
            field.state_path.assign(
                out,
                carrier.allocation.copy_state(field.state_path(source)),
            )
        return out

    def allocate_translation(self) -> Any:
        translation = self.translation_type(
            frame=self.frame,
            carriers=self.carriers,
            allocator=self,
            linear_combine=self.linear_combine,
            apply_translation=self.apply_translation,
            norm_kernel=self.norm,
        )
        for field, carrier in zip(self.frame.fields, self.carriers, strict=True):
            field.translation_path.assign(translation, carrier.allocation.zero_translation())
        return translation


__all__ = ["Allocator"]
