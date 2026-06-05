from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cupy as cp

from stark.accelerators import AcceleratorNone
from stark.algebraist.layout import AlgebraistLayout, AlgebraistLayoutLooped
from stark.algebraist.runtime import AlgebraistRuntimeGeneral, AlgebraistRuntimeSpecialist
from stark.carriers.cupy import CarrierCupy
from stark.contracts.accelerator import Accelerator
from stark.engines.cupy.allocator import StarkEngineAllocatorCupy
from stark.interface.layout import StarkLayout


@dataclass(frozen=True, slots=True)
class StarkEngineCupy:
    """CuPy-backed engine for a shaped `StarkLayout`."""

    layout: StarkLayout
    dtype: Any = cp.float64
    accelerator: Accelerator = field(default_factory=AcceleratorNone)
    algebraist_layout: AlgebraistLayout = field(init=False)
    carriers: tuple[CarrierCupy, ...] = field(init=False, repr=False)
    allocator: StarkEngineAllocatorCupy = field(init=False, repr=False)
    algebraist_general: AlgebraistRuntimeGeneral = field(init=False, repr=False)
    algebraist_specialist: AlgebraistRuntimeSpecialist = field(init=False, repr=False)

    def __post_init__(self) -> None:
        algebraist_layout = self.layout.to_algebraist_layout()
        carriers: list[CarrierCupy] = []

        for field in algebraist_layout.fields:
            policy = field.policy
            if not isinstance(policy, AlgebraistLayoutLooped) or policy.shape is None:
                raise ValueError(
                    "StarkEngineCupy requires every layout field to declare shape."
                )
            carriers.append(CarrierCupy(cp.zeros(policy.shape, dtype=self.dtype)))

        carrier_tuple = tuple(carriers)
        allocator = StarkEngineAllocatorCupy(
            algebraist_layout=algebraist_layout,
            carriers=carrier_tuple,
        )
        object.__setattr__(self, "algebraist_layout", algebraist_layout)
        object.__setattr__(self, "carriers", carrier_tuple)
        object.__setattr__(self, "allocator", allocator)
        object.__setattr__(
            self,
            "algebraist_general",
            AlgebraistRuntimeGeneral(
                translation=allocator.allocate_translation(),
                allocator=allocator,
                layout=algebraist_layout,
                accelerator=self.accelerator,
            ),
        )
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


__all__ = ["StarkEngineCupy"]
