from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, cast

from stark.core.contracts.accelerator import Accelerator
from stark.core.contracts.frame import FrameLike
from stark.core.contracts.state import State
from stark.core.contracts.translation import Translation
from stark.engines.accelerators.none import AcceleratorNone
from stark.engines._algebraist.allocator import AlgebraistAllocator
from stark.engines._algebraist.arity import AlgebraistArity
from stark.engines._algebraist.generator import (
    AlgebraistGeneratorInnerProduct,
    AlgebraistGeneratorLinearCombine,
    AlgebraistGeneratorNorm,
    AlgebraistGeneratorLinearFixed,
    AlgebraistGeneratorTarget,
    AlgebraistGeneratorTargetMutable,
)
from stark.engines._algebraist.inner_product import AlgebraistInnerProduct
from stark.engines._algebraist.linear_combine import AlgebraistLinearCombine
from stark.engines._algebraist.norm import AlgebraistNorm
from stark.engines._algebraist.linear_fixed import AlgebraistLinearFixed

StateType = TypeVar("StateType", bound=State)
TranslationType = TypeVar("TranslationType", bound=Translation)


class AlgebraistAllocatorHooks(Protocol[TranslationType]):
    linear_combine: tuple[Callable[..., TranslationType], ...]
    apply_translation: Callable[[object, TranslationType, object], object] | None
    norm: Callable[[TranslationType], float] | None
    inner_product: Callable[[TranslationType, TranslationType], float] | None


@dataclass(frozen=True, slots=True)
class Algebraist(Generic[StateType, TranslationType]):
    """Prepared algebra providers for one state/translation family."""

    linear_combine: AlgebraistLinearCombine[TranslationType]
    linear_fixed: AlgebraistLinearFixed[StateType, TranslationType]
    norm: AlgebraistNorm[TranslationType]
    inner_product: AlgebraistInnerProduct[TranslationType]

    @classmethod
    def generator(
        cls,
        *,
        frame: FrameLike,
        allocator: AlgebraistAllocator[TranslationType],
        accelerator: Accelerator | None = None,
        target: AlgebraistGeneratorTarget | None = None,
        max_arity: int = 12,
    ) -> "Algebraist[StateType, TranslationType]":
        """Prepare generated algebra for a known frame-backed layout."""

        if max_arity < 1:
            raise ValueError("max_arity must be at least 1.")

        resolved_accelerator = accelerator if accelerator is not None else AcceleratorNone()
        resolved_target = target if target is not None else AlgebraistGeneratorTargetMutable()
        translation = allocator.allocate_translation()

        linear_combine = AlgebraistGeneratorLinearCombine(
            translation=translation,
            allocator=allocator,
            frame=frame,
            accelerator=resolved_accelerator,
            target=resolved_target,
        )
        linear_fixed = AlgebraistGeneratorLinearFixed(
            translation=translation,
            allocator=allocator,
            frame=frame,
            accelerator=resolved_accelerator,
            target=resolved_target,
        )
        norm = AlgebraistGeneratorNorm(
            translation=translation,
            frame=frame,
            accelerator=resolved_accelerator,
            target=resolved_target,
        )
        inner_product = AlgebraistGeneratorInnerProduct(
            translation=translation,
            frame=frame,
            accelerator=resolved_accelerator,
            target=resolved_target,
        )

        hooks = cast(AlgebraistAllocatorHooks[TranslationType], allocator)
        hooks.linear_combine = tuple(
            linear_combine.provide(AlgebraistArity(arity))
            for arity in range(1, max_arity + 1)
        )
        hooks.apply_translation = linear_fixed.provide_unit_apply()
        hooks.norm = norm.provide()
        hooks.inner_product = inner_product.provide()

        return cls(
            linear_combine=linear_combine,
            linear_fixed=linear_fixed,
            norm=norm,
            inner_product=inner_product,
        )


__all__ = ["Algebraist"]
