from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from stark.core.contracts.accelerator import Accelerator
from stark.core.contracts.frame import FrameLike
from stark.core.contracts.state import State
from stark.core.contracts.translation import Translation
from stark.engines.accelerators.none import AcceleratorNone
from stark.engines.algebraist.allocator import AlgebraistAllocator
from stark.engines.algebraist.arity import AlgebraistArity
from stark.engines.algebraist.generator import (
    AlgebraistGeneratorInnerProduct,
    AlgebraistGeneratorLinearCombine,
    AlgebraistGeneratorNorm,
    AlgebraistGeneratorSpecialist,
    AlgebraistGeneratorTarget,
    AlgebraistGeneratorTargetMutable,
)
from stark.engines.algebraist.inner_product import AlgebraistInnerProduct
from stark.engines.algebraist.linear_combine import AlgebraistLinearCombine
from stark.engines.algebraist.norm import AlgebraistNorm
from stark.engines.algebraist.specialist import AlgebraistSpecialist

StateType = TypeVar("StateType", bound=State)
TranslationType = TypeVar("TranslationType", bound=Translation)


@dataclass(frozen=True, slots=True)
class Algebraist(Generic[StateType, TranslationType]):
    """Prepared algebra providers for one state/translation family."""

    linear_combine: AlgebraistLinearCombine[TranslationType]
    specialist: AlgebraistSpecialist[StateType, TranslationType]
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
        specialist = AlgebraistGeneratorSpecialist(
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

        object.__setattr__(
            allocator,
            "linear_combine",
            tuple(
                linear_combine.provide(AlgebraistArity(arity))
                for arity in range(1, max_arity + 1)
            ),
        )
        object.__setattr__(allocator, "apply_translation", specialist.provide_unit_apply())
        object.__setattr__(allocator, "norm", norm.provide())
        object.__setattr__(allocator, "inner_product", inner_product.provide())

        return cls(
            linear_combine=linear_combine,
            specialist=specialist,
            norm=norm,
            inner_product=inner_product,
        )


__all__ = ["Algebraist"]
