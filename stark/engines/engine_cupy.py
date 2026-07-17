from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import cupy as cp

from stark.engines.accelerators import AcceleratorNone
from stark.engines.carrier_cupy import CarrierCupy
from stark.engines.generator import (
    Generator,
    GeneratorPolicy,
    GeneratorRequestApplyTranslation,
    GeneratorRequestInnerProduct,
    GeneratorRequestLinearCombineTable,
    GeneratorRequestNorm,
)
from stark.engines.translation_factory_cupy import TranslationFactoryCupy
from stark.core.contracts.accelerator import Accelerator
from stark.core.contracts.frame import FrameLike
from stark.engines.allocator import AllocatorCarried
from stark.engines.translation_basis import TranslationBasis


@dataclass(slots=True)
class EngineCupy:
    """
    CuPy backend bundle for a shaped `Frame`.

    An engine supplies the backend objects used when a `System` prepares an
    IVP: carrier templates for each frame field, an allocator for owned state
    and translation objects, generated algebra providers, and an accelerator.
    This engine keeps field storage on CuPy arrays and uses a CuPy-native
    generator target so scheme algebra is emitted as CuPy kernels instead of
    Python loops over GPU arrays.
    """

    frame: FrameLike
    dtype: Any = cp.float64
    accelerator: Accelerator = field(default_factory=AcceleratorNone)
    carriers: tuple[CarrierCupy, ...] = field(init=False, repr=False)
    allocator: AllocatorCarried[Any] = field(init=False, repr=False)
    generator: Generator[Any, Any] = field(init=False, repr=False)

    def __repr__(self) -> str:
        accelerator_name = getattr(self.accelerator, "name", str(self.accelerator))
        acceleration = f"accelerator={accelerator_name!r}"
        if accelerator_name == "none":
            acceleration += ", note='CuPy ElementwiseKernel target; no separate Python kernel accelerator'"
        return (
            f"{type(self).__name__}(frame={self.frame!r}, "
            f"dtype={self.dtype!r}, {acceleration})"
        )

    def translation_basis(self) -> TranslationBasis:
        """Return the coordinate basis for engine-owned translations.

        This is an inspection and dense-materialisation helper. Ordinary user
        solves should not need it, but dense inverters and diagnostic examples
        can use it instead of hand-writing translation-basis classes.
        """

        return TranslationBasis(self.frame, self.carriers)

    def __post_init__(self) -> None:
        carriers: list[CarrierCupy] = []

        for field in self.frame.fields:
            policy = field.policy
            shape = getattr(policy, "shape", None)
            if shape is None:
                shape = getattr(field, "shape", None)
            if getattr(policy, "kind", None) != "looped" or shape is None:
                raise ValueError(
                    "EngineCupy requires every frame field to declare shape."
                )
            carriers.append(CarrierCupy(cp.zeros(shape, dtype=self.dtype)))

        carrier_tuple = tuple(carriers)
        allocator: AllocatorCarried[Any] = AllocatorCarried(
            frame=self.frame,
            carriers=carrier_tuple,
            translation_type=TranslationFactoryCupy,
        )
        self.carriers = carrier_tuple
        self.allocator = allocator

        generator = Generator(
            frame=self.frame,
            accelerator=self.accelerator,
            policy=GeneratorPolicy(
                traversal="elementwise",
                expression="elementwise",
                scalar="item",
            ),
            allocator=allocator,
        )
        allocator.apply_translation = cast(
            Callable[[Any, Any, Any], Any],
            generator(GeneratorRequestApplyTranslation()),
        )
        allocator.linear_combine = cast(
            tuple[Callable[..., Any], ...],
            generator(GeneratorRequestLinearCombineTable(max_arity=12)),
        )
        allocator.norm = cast(
            Callable[[Any], float],
            generator(GeneratorRequestNorm()),
        )
        allocator.inner_product = cast(
            Callable[[Any, Any], float],
            generator(GeneratorRequestInnerProduct()),
        )
        self.generator = generator


__all__ = ["EngineCupy"]
