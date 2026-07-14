from __future__ import annotations

from dataclasses import dataclass

from stark.engines.carrier_native.list.storage import CarrierNativeListValue, CarrierStorageNativeList


@dataclass(frozen=True)
class CarrierAllocationNativeList:
    storage: CarrierStorageNativeList

    def zero_state(self) -> CarrierNativeListValue:
        return [0 for _ in range(self.storage.length)]

    def zero_translation(self) -> CarrierNativeListValue:
        return self.zero_state()

    def allocate_translation(self) -> CarrierNativeListValue:
        return self.zero_translation()

    def copy_state(self, value: CarrierNativeListValue) -> CarrierNativeListValue:
        return list(value)

    def copy_translation(self, value: CarrierNativeListValue) -> CarrierNativeListValue:
        return list(value)
