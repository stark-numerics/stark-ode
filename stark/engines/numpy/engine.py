from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stark.accelerators import AcceleratorNone
from stark.algebraist.generator import AlgebraistGeneratorGeneral, AlgebraistGeneratorSpecialist
from stark.algebraist.layout import AlgebraistLayout, AlgebraistLayoutLooped
from stark.carriers import CarrierNumpy
from stark.contracts.accelerator import Accelerator
from stark.engines.numpy.allocator import StarkEngineAllocatorNumpy
from stark.interface.layout import StarkLayout


@dataclass(frozen=True, slots=True)
class StarkEngineNumpy:
    """NumPy-backed engine for a shaped `StarkLayout`."""

    layout: StarkLayout
    dtype: Any = np.float64
    accelerator: Accelerator = field(default_factory=AcceleratorNone)
    algebraist_layout: AlgebraistLayout = field(init=False)
    carriers: tuple[CarrierNumpy, ...] = field(init=False, repr=False)
    allocator: StarkEngineAllocatorNumpy = field(init=False, repr=False)
    algebraist_general: AlgebraistGeneratorGeneral = field(init=False, repr=False)
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
        object.__setattr__(
            self,
            "algebraist_general",
            AlgebraistGeneratorGeneral(
                translation=allocator.allocate_translation(),
                allocator=allocator,
                layout=algebraist_layout,
                accelerator=self.accelerator,
            ),
        )
        object.__setattr__(
            self,
            "algebraist_specialist",
            AlgebraistGeneratorSpecialist(
                translation=allocator.allocate_translation(),
                allocator=allocator,
                layout=algebraist_layout,
                accelerator=self.accelerator,
            ),
        )


__all__ = ["StarkEngineNumpy"]
