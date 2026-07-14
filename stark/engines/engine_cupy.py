from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cupy as cp

from stark.engines.accelerators import AcceleratorNone
from stark.engines.algebraist import Algebraist
from stark.engines.algebraist.generator.target_cupy import AlgebraistGeneratorTargetCupy
from stark.engines.carrier_cupy import CarrierCupy
from stark.engines.translation_factory_cupy import TranslationFactoryCupy
from stark.core.contracts.accelerator import Accelerator
from stark.core.contracts.frame import FrameLike
from stark.engines.allocator import Allocator
from stark.engines.translation_basis import TranslationBasis


@dataclass(frozen=True, slots=True)
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
    allocator: Allocator = field(init=False, repr=False)
    algebraist: Algebraist[Any, Any] = field(init=False, repr=False)

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
        allocator = Allocator(
            frame=self.frame,
            carriers=carrier_tuple,
            translation_type=TranslationFactoryCupy,
        )
        object.__setattr__(self, "carriers", carrier_tuple)
        object.__setattr__(self, "allocator", allocator)

        cupy_target = AlgebraistGeneratorTargetCupy()
        algebraist = Algebraist.generator(
            frame=self.frame,
            allocator=allocator,
            accelerator=self.accelerator,
            target=cupy_target,
        )
        object.__setattr__(self, "algebraist", algebraist)


__all__ = ["EngineCupy"]
