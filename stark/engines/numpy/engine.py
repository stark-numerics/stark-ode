from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stark.accelerators import AcceleratorNone, AcceleratorNumba
from stark.algebraist.arity import AlgebraistArity
from stark.algebraist.generator import (
    AlgebraistGeneratorLinearCombine,
    AlgebraistGeneratorNorm,
    AlgebraistGeneratorSpecialist,
)
from stark.algebraist.layout import AlgebraistLayout, AlgebraistLayoutLooped
from stark.carriers import CarrierNumpy
from stark.contracts.accelerator import Accelerator
from stark.engines.numpy.allocator import StarkEngineAllocatorNumpy
from stark.interface.layout import StarkLayout


def _default_accelerator() -> Accelerator:
    try:
        return AcceleratorNumba()
    except ModuleNotFoundError:
        return AcceleratorNone()


@dataclass(frozen=True, slots=True)
class StarkEngineNumpy:
    """
    NumPy backend bundle for a shaped `StarkLayout`.

    An engine supplies the backend objects used when a `StarkSystem` prepares an
    IVP: carrier templates for each layout field, an allocator for owned state
    and translation objects, the derived algebraist layout, generated algebra
    kernels, and an accelerator. By default this engine uses Numba when it is
    installed and otherwise falls back to unaccelerated NumPy-compatible
    callables.
    """

    layout: StarkLayout
    dtype: Any = np.float64
    accelerator: Accelerator = field(default_factory=_default_accelerator)
    algebraist_layout: AlgebraistLayout = field(init=False)
    carriers: tuple[CarrierNumpy, ...] = field(init=False, repr=False)
    allocator: StarkEngineAllocatorNumpy = field(init=False, repr=False)
    algebraist_linear_combine: AlgebraistGeneratorLinearCombine = field(init=False, repr=False)
    algebraist_norm: AlgebraistGeneratorNorm = field(init=False, repr=False)
    algebraist_specialist: AlgebraistGeneratorSpecialist = field(init=False, repr=False)

    def __post_init__(self) -> None:
        algebraist_layout = self.layout.to_algebraist_layout()
        carriers: list[CarrierNumpy] = []
        dtype = np.dtype(self.dtype)

        for field in algebraist_layout.fields:
            policy = field.policy
            if not isinstance(policy, AlgebraistLayoutLooped) or policy.shape is None:
                raise ValueError(
                    "StarkEngineNumpy requires every layout field to declare shape."
                )
            carriers.append(CarrierNumpy(np.zeros(policy.shape, dtype=dtype)))

        carrier_tuple = tuple(carriers)
        allocator = StarkEngineAllocatorNumpy(
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

        object.__setattr__(self, "algebraist_linear_combine", algebraist_linear_combine)
        object.__setattr__(self, "algebraist_norm", algebraist_norm)
        object.__setattr__(self, "algebraist_specialist", algebraist_specialist)


__all__ = ["StarkEngineNumpy"]
