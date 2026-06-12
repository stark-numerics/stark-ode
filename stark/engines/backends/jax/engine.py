from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp  # type: ignore[import-not-found]

from stark.engines.accelerators import AcceleratorJax
from stark.engines.algebraist.frame import (
    AlgebraistFrame,
    AlgebraistFrameLooped,
    AlgebraistFrameNormMax,
    AlgebraistFrameNormRMS,
)
from stark.engines.algebraist.runtime import (
    AlgebraistRuntimeInnerProduct,
    AlgebraistRuntimeLinearCombine,
    AlgebraistRuntimeNorm,
    AlgebraistRuntimeSpecialist,
)
from stark.engines.carriers.jax import CarrierJax, CarrierNormJaxMax, CarrierNormJaxRMS
from stark.contracts.accelerator import Accelerator
from stark.engines.backends.jax.allocator import EngineAllocatorJax
from stark.problem.frame.frame import Frame


@dataclass(frozen=True, slots=True)
class EngineJax:
    """
    JAX backend bundle for a shaped `Frame`.

    An engine supplies the backend objects used when a `System` prepares an
    IVP: carrier templates for each frame field, an allocator for owned state
    and translation objects, the derived algebraist frame, runtime algebra
    providers, and an accelerator. This engine keeps field storage in JAX arrays
    and exposes JAX-oriented algebra and derivative acceleration.
    """

    frame: Frame
    dtype: Any = jnp.float64
    accelerator: Accelerator = field(default_factory=AcceleratorJax)
    algebraist_frame: AlgebraistFrame = field(init=False)
    carriers: tuple[CarrierJax, ...] = field(init=False, repr=False)
    allocator: EngineAllocatorJax = field(init=False, repr=False)
    algebraist_inner_product: AlgebraistRuntimeInnerProduct = field(init=False, repr=False)
    algebraist_linear_combine: AlgebraistRuntimeLinearCombine = field(init=False, repr=False)
    algebraist_norm: AlgebraistRuntimeNorm = field(init=False, repr=False)
    algebraist_specialist: AlgebraistRuntimeSpecialist = field(init=False, repr=False)

    def __repr__(self) -> str:
        accelerator_name = getattr(self.accelerator, "name", str(self.accelerator))
        return (
            f"{type(self).__name__}(frame={self.frame!r}, "
            f"dtype={self.dtype!r}, accelerator={accelerator_name!r})"
        )

    def __post_init__(self) -> None:
        algebraist_frame = self.frame.to_algebraist_frame()
        carriers: list[CarrierJax] = []

        for field in algebraist_frame.fields:
            policy = field.policy
            if not isinstance(policy, AlgebraistFrameLooped) or policy.shape is None:
                raise ValueError(
                    "EngineJax requires every frame field to declare shape."
                )
            carriers.append(CarrierJax(jnp.zeros(policy.shape, dtype=self.dtype)))

        carrier_tuple = tuple(carriers)
        allocator = EngineAllocatorJax(
            algebraist_frame=algebraist_frame,
            carriers=carrier_tuple,
        )

        object.__setattr__(self, "algebraist_frame", algebraist_frame)
        object.__setattr__(self, "carriers", carrier_tuple)
        object.__setattr__(self, "allocator", allocator)
        field_norms = []
        for field, carrier in zip(algebraist_frame.fields, carrier_tuple, strict=True):
            if isinstance(field.norm, AlgebraistFrameNormRMS):
                field_norms.append(CarrierNormJaxRMS())
                continue
            if isinstance(field.norm, AlgebraistFrameNormMax):
                field_norms.append(CarrierNormJaxMax())
                continue
            if not field.norm.include:
                field_norms.append(carrier.norm)
                continue
            raise ValueError("EngineJax requires RMS, max, or excluded norm fields.")
        algebraist_norm = AlgebraistRuntimeNorm(
            frame=algebraist_frame,
            field_norms=tuple(field_norms),
        )
        object.__setattr__(allocator, "norm", algebraist_norm.provide())
        algebraist_inner_product = AlgebraistRuntimeInnerProduct(
            frame=algebraist_frame,
        )
        object.__setattr__(allocator, "inner_product", algebraist_inner_product.provide())
        object.__setattr__(
            self,
            "algebraist_linear_combine",
            AlgebraistRuntimeLinearCombine(
                translation=allocator.allocate_translation(),
                allocator=allocator,
                frame=algebraist_frame,
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
                frame=algebraist_frame,
                accelerator=self.accelerator,
            ),
        )


__all__ = ["EngineJax"]
