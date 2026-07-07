from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cupy as cp

from stark.engines.shared.accelerators import AcceleratorNone
from stark.engines.shared.algebraist.arity import AlgebraistArity
from stark.engines.shared.algebraist.generator import (
    AlgebraistGeneratorInnerProduct,
    AlgebraistGeneratorLinearCombine,
    AlgebraistGeneratorNorm,
    AlgebraistGeneratorSpecialist,
)
from stark.engines.cupy.target import AlgebraistGeneratorTargetCupy
from stark.engines.cupy.carriers import CarrierCupy
from stark.core.contracts.accelerator import Accelerator
from stark.core.contracts.frame import FrameLike
from stark.engines.cupy.allocator import EngineAllocatorCupy
from stark.engines.shared.basis import EngineTranslationBasis


@dataclass(frozen=True, slots=True)
class EngineCupy:
    """
    CuPy backend bundle for a shaped `Frame`.

    An engine supplies the backend objects used when a `System` prepares an
    IVP: carrier templates for each frame field, an allocator for owned state
    and translation objects, the derived algebraist frame, generated algebra
    providers, and an accelerator. This engine keeps field storage on CuPy
    arrays and uses a CuPy-native generator target so scheme algebra is emitted
    as CuPy kernels instead of Python loops over GPU arrays.
    """

    frame: FrameLike
    dtype: Any = cp.float64
    accelerator: Accelerator = field(default_factory=AcceleratorNone)
    algebraist_frame: FrameLike = field(init=False)
    carriers: tuple[CarrierCupy, ...] = field(init=False, repr=False)
    allocator: EngineAllocatorCupy = field(init=False, repr=False)
    algebraist_inner_product: AlgebraistGeneratorInnerProduct = field(init=False, repr=False)
    algebraist_linear_combine: AlgebraistGeneratorLinearCombine = field(init=False, repr=False)
    algebraist_norm: AlgebraistGeneratorNorm = field(init=False, repr=False)
    algebraist_specialist: AlgebraistGeneratorSpecialist = field(init=False, repr=False)

    def __repr__(self) -> str:
        accelerator_name = getattr(self.accelerator, "name", str(self.accelerator))
        acceleration = f"accelerator={accelerator_name!r}"
        if accelerator_name == "none":
            acceleration += ", note='CuPy ElementwiseKernel target; no separate Python kernel accelerator'"
        return (
            f"{type(self).__name__}(frame={self.frame!r}, "
            f"dtype={self.dtype!r}, {acceleration})"
        )

    def translation_basis(self) -> EngineTranslationBasis:
        """Return the coordinate basis for engine-owned translations.

        This is an inspection and dense-materialisation helper. Ordinary user
        solves should not need it, but dense inverters and diagnostic examples
        can use it instead of hand-writing translation-basis classes.
        """

        return EngineTranslationBasis(self.algebraist_frame, self.carriers)

    def __post_init__(self) -> None:
        algebraist_frame = self.frame
        carriers: list[CarrierCupy] = []

        for field in algebraist_frame.fields:
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
        allocator = EngineAllocatorCupy(
            algebraist_frame=algebraist_frame,
            carriers=carrier_tuple,
        )
        object.__setattr__(self, "algebraist_frame", algebraist_frame)
        object.__setattr__(self, "carriers", carrier_tuple)
        object.__setattr__(self, "allocator", allocator)

        cupy_target = AlgebraistGeneratorTargetCupy()
        algebraist_linear_combine = AlgebraistGeneratorLinearCombine(
            translation=allocator.allocate_translation(),
            allocator=allocator,
            frame=algebraist_frame,
            accelerator=self.accelerator,
            target=cupy_target,
        )
        object.__setattr__(
            allocator,
            "linear_combine",
            tuple(
                algebraist_linear_combine.provide(AlgebraistArity(arity))
                for arity in range(1, 13)
            ),
        )

        algebraist_specialist = AlgebraistGeneratorSpecialist(
            translation=allocator.allocate_translation(),
            allocator=allocator,
            frame=algebraist_frame,
            accelerator=self.accelerator,
            target=cupy_target,
        )
        object.__setattr__(
            allocator,
            "apply_translation",
            algebraist_specialist.provide_unit_apply(),
        )

        algebraist_norm = AlgebraistGeneratorNorm(
            translation=allocator.allocate_translation(),
            frame=algebraist_frame,
            accelerator=self.accelerator,
            target=cupy_target,
        )
        object.__setattr__(allocator, "norm", algebraist_norm.provide())

        algebraist_inner_product = AlgebraistGeneratorInnerProduct(
            translation=allocator.allocate_translation(),
            frame=algebraist_frame,
            accelerator=self.accelerator,
            target=cupy_target,
        )
        object.__setattr__(allocator, "inner_product", algebraist_inner_product.provide())

        object.__setattr__(self, "algebraist_linear_combine", algebraist_linear_combine)
        object.__setattr__(self, "algebraist_inner_product", algebraist_inner_product)
        object.__setattr__(self, "algebraist_norm", algebraist_norm)
        object.__setattr__(self, "algebraist_specialist", algebraist_specialist)


__all__ = ["EngineCupy"]
