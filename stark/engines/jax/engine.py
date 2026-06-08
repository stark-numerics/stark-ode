from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp  # type: ignore[import-not-found]

from stark.accelerators import AcceleratorJax
from stark.algebraist.layout import (
    AlgebraistLayout,
    AlgebraistLayoutLooped,
    AlgebraistLayoutNormMax,
    AlgebraistLayoutNormRMS,
)
from stark.algebraist.runtime import (
    AlgebraistRuntimeLinearCombine,
    AlgebraistRuntimeNorm,
    AlgebraistRuntimeSpecialist,
)
from stark.carriers.jax import CarrierJax, CarrierNormJaxMax, CarrierNormJaxRMS
from stark.contracts.accelerator import Accelerator
from stark.engines.jax.allocator import StarkEngineAllocatorJax
from stark.interface.layout import StarkLayout


@dataclass(frozen=True, slots=True)
class StarkEngineJax:
    """
    JAX backend bundle for a shaped `StarkLayout`.

    An engine supplies the backend objects used when a `StarkSystem` prepares an
    IVP: carrier templates for each layout field, an allocator for owned state
    and translation objects, the derived algebraist layout, runtime algebra
    providers, and an accelerator. This engine keeps field storage in JAX arrays
    and exposes JAX-oriented algebra and derivative acceleration.
    """

    layout: StarkLayout
    dtype: Any = jnp.float64
    accelerator: Accelerator = field(default_factory=AcceleratorJax)
    algebraist_layout: AlgebraistLayout = field(init=False)
    carriers: tuple[CarrierJax, ...] = field(init=False, repr=False)
    allocator: StarkEngineAllocatorJax = field(init=False, repr=False)
    algebraist_linear_combine: AlgebraistRuntimeLinearCombine = field(init=False, repr=False)
    algebraist_norm: AlgebraistRuntimeNorm = field(init=False, repr=False)
    algebraist_specialist: AlgebraistRuntimeSpecialist = field(init=False, repr=False)

    def __post_init__(self) -> None:
        algebraist_layout = self.layout.to_algebraist_layout()
        carriers: list[CarrierJax] = []

        for field in algebraist_layout.fields:
            policy = field.policy
            if not isinstance(policy, AlgebraistLayoutLooped) or policy.shape is None:
                raise ValueError(
                    "StarkEngineJax requires every layout field to declare shape."
                )
            carriers.append(CarrierJax(jnp.zeros(policy.shape, dtype=self.dtype)))

        carrier_tuple = tuple(carriers)
        allocator = StarkEngineAllocatorJax(
            algebraist_layout=algebraist_layout,
            carriers=carrier_tuple,
        )

        object.__setattr__(self, "algebraist_layout", algebraist_layout)
        object.__setattr__(self, "carriers", carrier_tuple)
        object.__setattr__(self, "allocator", allocator)
        field_norms = []
        for field, carrier in zip(algebraist_layout.fields, carrier_tuple, strict=True):
            if isinstance(field.norm, AlgebraistLayoutNormRMS):
                field_norms.append(CarrierNormJaxRMS())
                continue
            if isinstance(field.norm, AlgebraistLayoutNormMax):
                field_norms.append(CarrierNormJaxMax())
                continue
            if not field.norm.include:
                field_norms.append(carrier.norm)
                continue
            raise ValueError("StarkEngineJax requires RMS, max, or excluded norm fields.")
        algebraist_norm = AlgebraistRuntimeNorm(
            layout=algebraist_layout,
            field_norms=tuple(field_norms),
        )
        object.__setattr__(allocator, "norm", algebraist_norm.provide())
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


__all__ = ["StarkEngineJax"]
