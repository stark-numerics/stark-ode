from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from stark.core.contracts.engines.carrier import CarrierLike
from stark.core.contracts.problem.frame import FrameLike
from stark.engines.allocator.allocator import Allocator
from stark.engines.engine_translation import EngineTranslation


@Allocator.runtime
@dataclass(slots=True)
class EngineAllocator:
    """
    Allocate engine-owned state and translation objects for one frame.

    A concrete engine supplies one carrier per frame field. Each carrier knows
    how to allocate and copy the storage for that individual field; this
    allocator stitches those field values into the structured state and
    translation objects named by the frame paths.

    Engines may provide optimized translation hooks for common operations such
    as linear combinations, applying a translation to a state, norms, and inner
    products. New translations receive those hooks at allocation time, so
    scheme and solver code can use the same translation interface regardless of
    whether the hooks are generated or handwritten.

    `carriers` is intentionally typed at the heterogeneous boundary: different
    fields may store different concrete field-value types, while the allocator
    returns one engine-owned state object and one `EngineTranslation`.
    """

    frame: FrameLike
    carriers: tuple[CarrierLike[Any, Any], ...] = field(repr=False)
    linear_combine: tuple[Callable[..., EngineTranslation], ...] = field(
        default=(),
        repr=False,
    )
    apply_translation: Callable[[object, EngineTranslation, object], object] | None = (
        field(default=None, repr=False)
    )
    norm: Callable[[EngineTranslation], float] | None = field(default=None, repr=False)
    inner_product: Callable[[EngineTranslation, EngineTranslation], float] | None = field(
        default=None,
        repr=False,
    )

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

    def allocate_translation(self) -> EngineTranslation:
        translation = EngineTranslation(
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


__all__ = ["EngineAllocator"]
