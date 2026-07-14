from __future__ import annotations

from array import array
from dataclasses import dataclass

from stark.engines.carrier_native.array.storage import CarrierNativeArrayValue, CarrierStorageNativeArray


@dataclass(frozen=True)
class CarrierAllocationNativeArray:
    storage: CarrierStorageNativeArray

    def zero_state(self) -> CarrierNativeArrayValue:
        return array(self.storage.typecode, (0.0 for _ in range(self.storage.length)))

    def zero_translation(self) -> CarrierNativeArrayValue:
        return self.zero_state()

    def allocate_translation(self) -> CarrierNativeArrayValue:
        return self.zero_translation()

    def copy_state(self, value: CarrierNativeArrayValue) -> CarrierNativeArrayValue:
        return array(value.typecode, value)

    def copy_translation(self, value: CarrierNativeArrayValue) -> CarrierNativeArrayValue:
        return array(value.typecode, value)
