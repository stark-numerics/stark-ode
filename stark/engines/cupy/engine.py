from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cupy as cp

from stark.accelerators import AcceleratorNone
from stark.algebraist.layout import (
    AlgebraistLayout,
    AlgebraistLayoutLooped,
    AlgebraistLayoutNormMax,
    AlgebraistLayoutNormRMS,
)
from stark.algebraist.runtime import (
    AlgebraistRuntimeInnerProduct,
    AlgebraistRuntimeLinearCombine,
    AlgebraistRuntimeNorm,
    AlgebraistRuntimeSpecialist,
)
from stark.carriers.cupy import CarrierCupy, CarrierNormCupyMax, CarrierNormCupyRMS
from stark.contracts.accelerator import Accelerator
from stark.engines.cupy.allocator import EngineAllocatorCupy
from stark.interface.layout import Layout


@dataclass(frozen=True, slots=True)
class EngineCupy:
    """
    CuPy backend bundle for a shaped `Layout`.

    An engine supplies the backend objects used when a `System` prepares an
    IVP: carrier templates for each layout field, an allocator for owned state
    and translation objects, the derived algebraist layout, runtime algebra
    providers, and an accelerator. This engine keeps field storage on CuPy
    arrays and uses CuPy-native runtime algebra rather than generated Numba
    kernels.
    """

    layout: Layout
    dtype: Any = cp.float64
    accelerator: Accelerator = field(default_factory=AcceleratorNone)
    algebraist_layout: AlgebraistLayout = field(init=False)
    carriers: tuple[CarrierCupy, ...] = field(init=False, repr=False)
    allocator: EngineAllocatorCupy = field(init=False, repr=False)
    algebraist_inner_product: AlgebraistRuntimeInnerProduct = field(init=False, repr=False)
    algebraist_linear_combine: AlgebraistRuntimeLinearCombine = field(init=False, repr=False)
    algebraist_norm: AlgebraistRuntimeNorm = field(init=False, repr=False)
    algebraist_specialist: AlgebraistRuntimeSpecialist = field(init=False, repr=False)

    def __repr__(self) -> str:
        accelerator_name = getattr(self.accelerator, "name", str(self.accelerator))
        acceleration = f"accelerator={accelerator_name!r}"
        if accelerator_name == "none":
            acceleration += ", note='CuPy runtime algebra; no separate Python kernel accelerator'"
        return (
            f"{type(self).__name__}(layout={self.layout!r}, "
            f"dtype={self.dtype!r}, {acceleration})"
        )

    def __post_init__(self) -> None:
        algebraist_layout = self.layout.to_algebraist_layout()
        carriers: list[CarrierCupy] = []

        for field in algebraist_layout.fields:
            policy = field.policy
            if not isinstance(policy, AlgebraistLayoutLooped) or policy.shape is None:
                raise ValueError(
                    "EngineCupy requires every layout field to declare shape."
                )
            carriers.append(CarrierCupy(cp.zeros(policy.shape, dtype=self.dtype)))

        carrier_tuple = tuple(carriers)
        allocator = EngineAllocatorCupy(
            algebraist_layout=algebraist_layout,
            carriers=carrier_tuple,
        )
        object.__setattr__(self, "algebraist_layout", algebraist_layout)
        object.__setattr__(self, "carriers", carrier_tuple)
        object.__setattr__(self, "allocator", allocator)
        field_norms = []
        for field, carrier in zip(algebraist_layout.fields, carrier_tuple, strict=True):
            if isinstance(field.norm, AlgebraistLayoutNormRMS):
                field_norms.append(CarrierNormCupyRMS())
                continue
            if isinstance(field.norm, AlgebraistLayoutNormMax):
                field_norms.append(CarrierNormCupyMax())
                continue
            if not field.norm.include:
                field_norms.append(carrier.norm)
                continue
            raise ValueError("EngineCupy requires RMS, max, or excluded norm fields.")
        algebraist_norm = AlgebraistRuntimeNorm(
            layout=algebraist_layout,
            field_norms=tuple(field_norms),
        )
        object.__setattr__(allocator, "norm", algebraist_norm.provide())
        algebraist_inner_product = AlgebraistRuntimeInnerProduct(
            layout=algebraist_layout,
        )
        object.__setattr__(allocator, "inner_product", algebraist_inner_product.provide())
        object.__setattr__(
            self,
            "algebraist_linear_combine",
            AlgebraistRuntimeLinearCombine(
                translation=allocator.allocate_translation(),
                allocator=allocator,
                layout=algebraist_layout,
                accelerator=self.accelerator,
            ),
        )
        object.__setattr__(self, "algebraist_inner_product", algebraist_inner_product)
        object.__setattr__(self, "algebraist_norm", algebraist_norm)
        object.__setattr__(
            self,
            "algebraist_specialist",
            AlgebraistRuntimeSpecialist(
                translation=allocator.allocate_translation(),
                allocator=allocator,
                layout=algebraist_layout,
                accelerator=self.accelerator,
            ),
        )


__all__ = ["EngineCupy"]
