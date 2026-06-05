from __future__ import annotations

from array import array
from dataclasses import dataclass, field

from stark.accelerators import AcceleratorNone
from stark.algebraist.generator import AlgebraistGeneratorGeneral, AlgebraistGeneratorSpecialist
from stark.algebraist.layout import AlgebraistLayout, AlgebraistLayoutLooped
from stark.carriers.native import CarrierNativeArray
from stark.contracts.accelerator import Accelerator
from stark.engines.native.allocator import StarkEngineAllocatorNative
from stark.interface.layout import StarkLayout


@dataclass(frozen=True, slots=True)
class StarkEngineNative:
    """Native Python engine for one-dimensional shaped layouts."""

    layout: StarkLayout
    typecode: str = "d"
    accelerator: Accelerator = field(default_factory=AcceleratorNone)
    algebraist_layout: AlgebraistLayout = field(init=False)
    carriers: tuple[CarrierNativeArray, ...] = field(init=False, repr=False)
    allocator: StarkEngineAllocatorNative = field(init=False, repr=False)
    algebraist_general: AlgebraistGeneratorGeneral = field(init=False, repr=False)
    algebraist_specialist: AlgebraistGeneratorSpecialist = field(init=False, repr=False)

    def __post_init__(self) -> None:
        algebraist_layout = self.layout.to_algebraist_layout()
        carriers: list[CarrierNativeArray] = []

        for field in algebraist_layout.fields:
            policy = field.policy
            if not isinstance(policy, AlgebraistLayoutLooped) or policy.shape is None:
                raise ValueError(
                    "StarkEngineNative requires every layout field to declare shape."
                )
            if len(policy.shape) != 1:
                raise ValueError(
                    "StarkEngineNative currently supports one-dimensional field shapes only."
                )
            carriers.append(CarrierNativeArray(array(self.typecode, (0.0 for _ in range(policy.shape[0])))))

        carrier_tuple = tuple(carriers)
        allocator = StarkEngineAllocatorNative(
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


__all__ = ["StarkEngineNative"]
