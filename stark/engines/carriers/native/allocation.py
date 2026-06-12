from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from stark.engines.carriers.native.array import CarrierAllocationNativeArray
from stark.engines.carriers.native.list import CarrierAllocationNativeList
from stark.engines.carriers.native.scalar import CarrierAllocationNativeScalar
from stark.engines.carriers.native.storage import CarrierNativeValue, CarrierStorageNative
from stark.engines.carriers.native.tuple import CarrierAllocationNativeTuple


@dataclass(frozen=True)
class CarrierAllocationNative:
    """Compatibility facade for code that imports the old native allocation class."""

    storage: CarrierStorageNative

    def __post_init__(self) -> None:
        concrete_storage = self.storage.concrete
        if concrete_storage.__class__.__name__ == "CarrierStorageNativeScalar":
            concrete = CarrierAllocationNativeScalar(concrete_storage)
        elif concrete_storage.__class__.__name__ == "CarrierStorageNativeList":
            concrete = CarrierAllocationNativeList(concrete_storage)
        elif concrete_storage.__class__.__name__ == "CarrierStorageNativeTuple":
            concrete = CarrierAllocationNativeTuple(concrete_storage)
        elif concrete_storage.__class__.__name__ == "CarrierStorageNativeArray":
            concrete = CarrierAllocationNativeArray(concrete_storage)
        else:  # pragma: no cover - defensive against future storage classes.
            raise TypeError("Unsupported native storage class.")
        object.__setattr__(self, "concrete", concrete)

    concrete: Any = field(init=False)

    def zero_state(self) -> CarrierNativeValue:
        return self.concrete.zero_state()

    def zero_translation(self) -> CarrierNativeValue:
        return self.concrete.zero_translation()

    def allocate_translation(self) -> CarrierNativeValue:
        return self.concrete.allocate_translation()

    def copy_state(self, value: CarrierNativeValue) -> CarrierNativeValue:
        return self.concrete.copy_state(value)

    def copy_translation(self, value: CarrierNativeValue) -> CarrierNativeValue:
        return self.concrete.copy_translation(value)
