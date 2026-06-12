from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stark.engines.accelerators import AcceleratorNone, AcceleratorNumba
from stark.engines.algebraist.arity import AlgebraistArity
from stark.engines.algebraist.generator import (
    AlgebraistGeneratorInnerProduct,
    AlgebraistGeneratorLinearCombine,
    AlgebraistGeneratorNorm,
    AlgebraistGeneratorSpecialist,
)
from stark.engines.algebraist.layout import AlgebraistLayout, AlgebraistLayoutLooped
from stark.engines.carriers import CarrierNumpy
from stark.contracts.accelerator import Accelerator
from stark.engines.backends.numpy.allocator import EngineAllocatorNumpy
from stark.interface.layout import Layout


def _default_accelerator() -> Accelerator:
    try:
        return AcceleratorNumba()
    except ModuleNotFoundError:
        return AcceleratorNone()


@dataclass(frozen=True, slots=True)
class EngineNumpy:
    """
    NumPy backend bundle for a shaped `Layout`.

    An engine supplies the backend objects used when a `System` prepares an
    IVP: carrier templates for each layout field, an allocator for owned state
    and translation objects, the derived algebraist layout, generated algebra
    kernels, and an accelerator. By default this engine uses Numba when it is
    installed and otherwise falls back to unaccelerated NumPy-compatible
    callables.
    """

    layout: Layout
    dtype: Any = np.float64
    accelerator: Accelerator = field(default_factory=_default_accelerator)
    algebraist_layout: AlgebraistLayout = field(init=False)
    carriers: tuple[CarrierNumpy, ...] = field(init=False, repr=False)
    allocator: EngineAllocatorNumpy = field(init=False, repr=False)
    algebraist_inner_product: AlgebraistGeneratorInnerProduct = field(init=False, repr=False)
    algebraist_linear_combine: AlgebraistGeneratorLinearCombine = field(init=False, repr=False)
    algebraist_norm: AlgebraistGeneratorNorm = field(init=False, repr=False)
    algebraist_specialist: AlgebraistGeneratorSpecialist = field(init=False, repr=False)

    def __repr__(self) -> str:
        accelerator_name = getattr(self.accelerator, "name", str(self.accelerator))
        acceleration = f"accelerator={accelerator_name!r}"
        if accelerator_name == "none":
            acceleration += (
                ", WARNING='unaccelerated CPU engine; install numba or pass an "
                "accelerator to compile generated kernels'"
            )
        return (
            f"{type(self).__name__}(layout={self.layout!r}, "
            f"dtype={np.dtype(self.dtype)!r}, {acceleration})"
        )

    def __post_init__(self) -> None:
        algebraist_layout = self.layout.to_algebraist_layout()
        carriers: list[CarrierNumpy] = []
        dtype = np.dtype(self.dtype)

        for field in algebraist_layout.fields:
            policy = field.policy
            if not isinstance(policy, AlgebraistLayoutLooped) or policy.shape is None:
                raise ValueError(
                    "EngineNumpy requires every layout field to declare shape."
                )
            carriers.append(CarrierNumpy(np.zeros(policy.shape, dtype=dtype)))

        carrier_tuple = tuple(carriers)
        allocator = EngineAllocatorNumpy(
            algebraist_layout=algebraist_layout,
            carriers=carrier_tuple,
        )

        object.__setattr__(self, "algebraist_layout", algebraist_layout)
        object.__setattr__(self, "carriers", carrier_tuple)
        object.__setattr__(self, "allocator", allocator)
        algebraist_linear_combine = AlgebraistGeneratorLinearCombine(
            translation=allocator.allocate_translation(),
            allocator=allocator,
            layout=algebraist_layout,
            accelerator=self.accelerator,
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
            layout=algebraist_layout,
            accelerator=self.accelerator,
        )
        object.__setattr__(
            allocator,
            "apply_translation",
            algebraist_specialist.provide_unit_apply(),
        )
        algebraist_norm = AlgebraistGeneratorNorm(
            translation=allocator.allocate_translation(),
            layout=algebraist_layout,
            accelerator=self.accelerator,
        )
        object.__setattr__(allocator, "norm", algebraist_norm.provide())
        algebraist_inner_product = AlgebraistGeneratorInnerProduct(
            translation=allocator.allocate_translation(),
            layout=algebraist_layout,
            accelerator=self.accelerator,
        )
        object.__setattr__(allocator, "inner_product", algebraist_inner_product.provide())

        object.__setattr__(self, "algebraist_inner_product", algebraist_inner_product)
        object.__setattr__(self, "algebraist_linear_combine", algebraist_linear_combine)
        object.__setattr__(self, "algebraist_norm", algebraist_norm)
        object.__setattr__(self, "algebraist_specialist", algebraist_specialist)


__all__ = ["EngineNumpy"]
